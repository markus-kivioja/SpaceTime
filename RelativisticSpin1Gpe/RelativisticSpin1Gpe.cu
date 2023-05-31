#include <cuda_runtime.h>
#include "helper_cuda.h"

//#include "AliceRingRamps.h"
#include "KnotRamps.h"

std::string getProjectionString()
{
#if BASIS == X_QUANTIZED
	return "proj_x";
#elif BASIS == Y_QUANTIZED
	return "proj_y";
#elif BASIS == Z_QUANTIZED
	return "proj_z";
#endif
}

#include "Output/Picture.hpp"
#include "Output/Text.hpp"
#include "Types/Complex.hpp"
#include "Mesh/DelaunayMesh.hpp"

#include <iostream>
#include <sstream>
#include <chrono>
#include <random>

#include "mesh.h"

#define COMPUTE_GROUND_STATE 0

#define HYPERBOLIC 0
#define PARABOLIC 1
#define COMPUTE_ERROR (HYPERBOLIC && PARABOLIC)

#define SAVE_STATES 0
#define SAVE_PICTURE 1

#define THREAD_BLOCK_X 16
#define THREAD_BLOCK_Y 2
#define THREAD_BLOCK_Z 1

constexpr double DOMAIN_SIZE_X = 16.0;
constexpr double DOMAIN_SIZE_Y = 16.0;
constexpr double DOMAIN_SIZE_Z = 16.0;

constexpr double REPLICABLE_STRUCTURE_COUNT_X = 58.0 + 9 * 6.0;
//constexpr double REPLICABLE_STRUCTURE_COUNT_Y = 112.0;
//constexpr double REPLICABLE_STRUCTURE_COUNT_Z = 112.0;

constexpr double N = 2e5; // Number of atoms in the condensate

constexpr double trapFreq_r = 126;
constexpr double trapFreq_z = 166;

constexpr double omega_r = trapFreq_r * 2 * PI;
constexpr double omega_z = trapFreq_z * 2 * PI;
constexpr double lambda_x = 1.0;
constexpr double lambda_y = 1.0;
constexpr double lambda_z = omega_z / omega_r;

constexpr double a_bohr = 5.2917721092e-11; //[m] Bohr radius
constexpr double a_0 = 101.8;
constexpr double a_2 = 100.4;

constexpr double atomMass = 1.44316060e-25;
constexpr double hbar = 1.05457148e-34; // [m^2 kg / s]
const double a_r = sqrt(hbar / (atomMass * omega_r)); //[m]

const double c0 = 4 * PI * N * (a_0 + 2 * a_2) * a_bohr / (3 * a_r);
const double c2 = 4 * PI * N * (a_2 - a_0) * a_bohr / (3 * a_r);

constexpr double muB = 9.27400968e-24; // [m^2 kg / s^2 T^-1] Bohr magneton

const double BqScale = -(0.5 * muB / (hbar * omega_r) * a_r) / 100.; // [cm/Gauss]
constexpr double BzScale = -(0.5 * muB / (hbar * omega_r)) / 10000.; // [1/Gauss]

constexpr double A_hfs = 3.41734130545215;
const double BqQuadScale = 100 * a_r * sqrt(0.25 * 1000 * (1.399624624 * 1.399624624) / (trapFreq_r * 2 * A_hfs)); //[cm/Gauss]
const double BzQuadScale = sqrt(0.25 * 1000 * (1.399624624 * 1.399624624) / (trapFreq_r * 2 * A_hfs)); //[1/Gauss]  \sqrt{g_q}

constexpr double SQRT_2 = 1.41421356237309;
constexpr double INV_SQRT_2 = 0.70710678118655;

constexpr double NOISE_AMPLITUDE = 0.1;

double dt = 3.9e-4;
double dt_increse = 1e-5;

const float IMAGE_SAVE_INTERVAL = 0.05; // ms
uint IMAGE_SAVE_FREQUENCY = uint(IMAGE_SAVE_INTERVAL * 0.5 / 1e3 * omega_r / dt) + 1;

const uint STATE_SAVE_INTERVAL = 10.0; // ms

double t = 0; // Start time in ms
double END_TIME = 0.55; // End time in ms

#if COMPUTE_GROUND_STATE
double sigma = 0.1; // 0.01; // Coefficient for the relativistic term (zero for non-relativistic)
#else
double sigma = 1.0; // 0.01; // Coefficient for the relativistic term (zero for non-relativistic)
#endif
double dt_per_sigma = dt / sigma;

enum class Phase
{
	Polar = 0,
	Ferromagnetic,
	None
};
//constexpr Phase initPhase = Phase::None;
constexpr Phase initPhase = Phase::Polar;
//constexpr Phase initPhase = Phase::Ferromagnetic;

std::string toStringShort(const double value)
{
	std::ostringstream out;
	out.precision(2);
	out << std::fixed << value;
	return out.str();
};

std::string phaseToString(Phase phase)
{
	switch (phase)
	{
	case Phase::Polar:
		return "polar";
	case Phase::Ferromagnetic:
		return "ferromagnetic";
	case Phase::None:
		return "none";
	default:
		return "";
	}
}

const std::string EXTRA_INFORMATION = toStringShort(DOMAIN_SIZE_X) + "_" + toStringShort(REPLICABLE_STRUCTURE_COUNT_X) + "_" + phaseToString(initPhase);
const std::string GROUND_STATE_PSI_FILENAME = "ground_state_psi_" + EXTRA_INFORMATION + ".dat";
const std::string GROUND_STATE_Q_FILENAME = "ground_state_q_" + EXTRA_INFORMATION + ".dat";

#include "hyper_kernels.h"
#include "para_kernels.h"
#include "common_kernels.h"

void normalize_h(dim3 dimGrid, dim3 dimBlock, double* densityPtr, PitchedPtr psi, uint3 dimensions, size_t bodies, double volume)
{
	density << <dimGrid, dimBlock >> > (densityPtr, psi, dimensions);
	int prevStride = bodies;
	while (prevStride > 1)
	{
		int newStride = prevStride / 2;
		integrate << <dim3(std::ceil(newStride / 32.0), 1, 1), dim3(32, 1, 1) >> > (densityPtr, newStride, ((newStride * 2) != prevStride), volume);
		prevStride = newStride;
	}

	normalize << < dimGrid, dimBlock >> > (densityPtr, psi, dimensions);
}

double getDensity(dim3 dimGrid, dim3 dimBlock, double* densityPtr, PitchedPtr psi, uint3 dimensions, size_t bodies, double volume)
{
	density << <dimGrid, dimBlock >> > (densityPtr, psi, dimensions);
	int prevStride = bodies;
	while (prevStride > 1)
	{
		int newStride = prevStride / 2;
		integrate << <dim3(std::ceil(newStride / 32.0), 1, 1), dim3(32, 1, 1) >> > (densityPtr, newStride, ((newStride * 2) != prevStride), volume);
		prevStride = newStride;
	}
	double hDensity = 0;
	checkCudaErrors(cudaMemcpy(&hDensity, densityPtr, sizeof(double), cudaMemcpyDeviceToHost));

	return hDensity;
}

struct SpinMagDens
{
	double spin;
	double3 magnetization;
	double density;
};

SpinMagDens integrateSpinAndDensity(dim3 dimGrid, dim3 dimBlock, double* spinNormPtr, double3* localAvgSpinPtr, double* densityPtr, size_t bodies, double volume)
{
	int prevStride = bodies;
	while (prevStride > 1)
	{
		int newStride = prevStride / 2;
		if (prevStride == bodies)
		{
			integrateVecWithDensity << <dim3(std::ceil(newStride / 32.0), 1, 1), dim3(32, 1, 1) >> > (localAvgSpinPtr, densityPtr, newStride, ((newStride * 2) != prevStride), volume);
		}
		else
		{
			integrateVec << <dim3(std::ceil(newStride / 32.0), 1, 1), dim3(32, 1, 1) >> > (localAvgSpinPtr, newStride, ((newStride * 2) != prevStride), volume);
		}
		integrate << <dim3(std::ceil(newStride / 32.0), 1, 1), dim3(32, 1, 1) >> > (spinNormPtr, newStride, ((newStride * 2) != prevStride), volume);
		integrate << <dim3(std::ceil(newStride / 32.0), 1, 1), dim3(32, 1, 1) >> > (densityPtr, newStride, ((newStride * 2) != prevStride), volume);
		prevStride = newStride;
	}
	double3 hMagnetization = { 0, 0, 0 };
	checkCudaErrors(cudaMemcpy(&hMagnetization, localAvgSpinPtr, sizeof(double3), cudaMemcpyDeviceToHost));

	double hSpinNorm = 0;
	checkCudaErrors(cudaMemcpy(&hSpinNorm, spinNormPtr, sizeof(double), cudaMemcpyDeviceToHost));

	double hDensity = 0;
	checkCudaErrors(cudaMemcpy(&hDensity, densityPtr, sizeof(double), cudaMemcpyDeviceToHost));

	return { hSpinNorm, hMagnetization, hDensity };
}

float getMaxHamilton(dim3 dimGrid, dim3 dimBlock, double* maxHamlPtr, PitchedPtr psi, MagFields Bs, uint3 dimensions, size_t bodies, double block_scale, double3 p0)
{
	maxHamilton << <dimGrid, dimBlock >> > (maxHamlPtr, psi, Bs, dimensions, block_scale, p0, c0, c2);
	int prevStride = bodies;
	while (prevStride > 1)
	{
		int newStride = prevStride / 2;
		reduceMax << <dim3(std::ceil(newStride / 32.0), 1, 1), dim3(32, 1, 1) >> > (maxHamlPtr, newStride, ((newStride * 2) != prevStride));
		prevStride = newStride;
	}

	double maxHaml = 0;
	checkCudaErrors(cudaMemcpy(&maxHaml, maxHamlPtr, sizeof(double), cudaMemcpyDeviceToHost));

	return maxHaml;
}

template<typename T>
T* allocDevice(size_t count)
{
	T* ptr;
	checkCudaErrors(cudaMalloc(&ptr, count * sizeof(T)));
	return ptr;
}

cudaPitchedPtr allocDevice3D(cudaExtent extent)
{
	cudaPitchedPtr ptr;
	checkCudaErrors(cudaMalloc3D(&ptr, extent));
	return ptr;
}

template<typename T>
T* allocHost(size_t count)
{
	T* ptr;
	checkCudaErrors(cudaMallocHost(&ptr, count * sizeof(T)));
	memset(ptr, 0, count * sizeof(T));
	return ptr;
}

cudaPitchedPtr copyHostToDevice3D(void* src, cudaPitchedPtr dst, cudaExtent extent)
{
	cudaPitchedPtr cuda_src = { src, extent.width, dst.xsize, dst.ysize };

	cudaMemcpy3DParms params = { 0 };
	params.srcPtr = cuda_src;
	params.dstPtr = dst;
	params.extent = extent;
	params.kind = cudaMemcpyHostToDevice;

	checkCudaErrors(cudaMemcpy3D(&params));

	return cuda_src;
}

cudaMemcpy3DParms createDeviceToHostParams(cudaPitchedPtr src, cudaPitchedPtr dst, cudaExtent extent)
{
	cudaMemcpy3DParms params = { 0 };
	params.srcPtr = src;
	params.dstPtr = dst;
	params.extent = extent;
	params.kind = cudaMemcpyDeviceToHost;

	return params;
}

void loadFromFile(const std::string& filename, char* dst, size_t size)
{
	std::ifstream psi_fs(filename, std::ios::binary | std::ios::in);
	if (psi_fs.fail() != 0)
	{
		std::cout << "Failed to open file " << filename << std::endl;
		exit(1);
	}
	else
	{
		std::cout << "Loading data from " << filename << "..." << std::endl;
	}
	psi_fs.read(dst, size);
	psi_fs.close();
}

uint integrateInTime(const double block_scale, const Vector3& minp, const Vector3& maxp)
{
	// find dimensions
	const Vector3 domain = maxp - minp;
	const uint xsize = uint(domain.x / (block_scale * BLOCK_WIDTH.x)); // + 1;
	const uint ysize = uint(domain.y / (block_scale * BLOCK_WIDTH.y)); // + 1;
	const uint zsize = uint(domain.z / (block_scale * BLOCK_WIDTH.z)); // + 1;
	const Vector3 p0 = 0.5 * (minp + maxp - block_scale * Vector3(BLOCK_WIDTH.x * xsize, BLOCK_WIDTH.y * ysize, BLOCK_WIDTH.z * zsize));
	const double3 d_p0 = { p0.x, p0.y, p0.z };

	// compute discrete dimensions
	const uint bsize = VALUES_IN_BLOCK; // bpos.size(); // number of values inside a block

	//std::cout << "Dual 0-cells in a replicable structure: " << bsize << std::endl;
	std::cout << "Replicable structure instances in x: " << xsize << ", y: " << ysize << ", z: " << zsize << std::endl;
	uint64_t bodies = xsize * ysize * zsize * bsize;
	std::cout << "Dual 0-cells in total: " << bodies << std::endl;

	// Initialize device memory
	size_t dxsize = xsize + 2; // One element buffer to both ends
	size_t dysize = ysize + 2; // One element buffer to both ends
	size_t dzsize = zsize + 2; // One element buffer to both ends
	cudaExtent psiExtent = make_cudaExtent(dxsize * sizeof(BlockPsis), dysize, dzsize);
	cudaExtent edgeExtent = make_cudaExtent(dxsize * sizeof(BlockEdges), dysize, dzsize);

#if HYPERBOLIC
	cudaPitchedPtr d_cudaEvenPsiHyper = allocDevice3D(psiExtent);
	cudaPitchedPtr d_cudaEvenQHyper = allocDevice3D(edgeExtent);
	cudaPitchedPtr d_cudaOddPsiHyper = allocDevice3D(psiExtent);
	cudaPitchedPtr d_cudaOddQHyper = allocDevice3D(edgeExtent);
#endif
#if PARABOLIC
	cudaPitchedPtr d_cudaEvenPsiPara = allocDevice3D(psiExtent);
	cudaPitchedPtr d_cudaEvenQPara = allocDevice3D(edgeExtent);
	cudaPitchedPtr d_cudaOddPsiPara = allocDevice3D(psiExtent);
	cudaPitchedPtr d_cudaOddQPara = allocDevice3D(edgeExtent);
#endif

#if COMPUTE_ERROR
	//double2* d_error = allocDevice<double2>(bodies);
	double* d_error = allocDevice<double>(bodies);
#endif

#if COMPUTE_GROUND_STATE
	// For computing the energy/chemical potential
	cudaPitchedPtr d_cudaHPsi = allocDevice3D(edgeExtent);
#endif

	//double* d_spinNorm = allocDevice<double>(bodies);
	double* d_density = allocDevice<double>(bodies);
#if COMPUTE_GROUND_STATE
	double* d_energy = allocDevice<double>(bodies);
#endif

	// Calculate pointers to the start of the real computational domain (jumping over the zero buffer at the edges)
#if HYPERBOLIC
	size_t offsetHyper = d_cudaEvenPsiHyper.pitch * dysize + d_cudaEvenPsiHyper.pitch + sizeof(BlockPsis);
	size_t edgeOffsetHyper = d_cudaEvenQHyper.pitch * dysize + d_cudaEvenQHyper.pitch + sizeof(BlockEdges);
	PitchedPtr d_evenPsiHyper = { (char*)d_cudaEvenPsiHyper.ptr + offsetHyper, d_cudaEvenPsiHyper.pitch, d_cudaEvenPsiHyper.pitch * dysize };
	PitchedPtr d_evenQHyper = { (char*)d_cudaEvenQHyper.ptr + edgeOffsetHyper, d_cudaEvenQHyper.pitch, d_cudaEvenQHyper.pitch * dysize };
	PitchedPtr d_oddPsiHyper = { (char*)d_cudaOddPsiHyper.ptr + offsetHyper, d_cudaOddPsiHyper.pitch, d_cudaOddPsiHyper.pitch * dysize };
	PitchedPtr d_oddQHyper = { (char*)d_cudaOddQHyper.ptr + edgeOffsetHyper, d_cudaOddQHyper.pitch, d_cudaOddQHyper.pitch * dysize };
#endif
#if PARABOLIC
	size_t offsetPara = d_cudaEvenPsiPara.pitch * dysize + d_cudaEvenPsiPara.pitch + sizeof(BlockPsis);
	size_t edgeOffsetPara = d_cudaEvenQPara.pitch * dysize + d_cudaEvenQPara.pitch + sizeof(BlockEdges);
	PitchedPtr d_evenPsiPara = { (char*)d_cudaEvenPsiPara.ptr + offsetPara, d_cudaEvenPsiPara.pitch, d_cudaEvenPsiPara.pitch * dysize };
	PitchedPtr d_evenQPara = { (char*)d_cudaEvenQPara.ptr + edgeOffsetPara, d_cudaEvenQPara.pitch, d_cudaEvenQPara.pitch * dysize };
	PitchedPtr d_oddPsiPara = { (char*)d_cudaOddPsiPara.ptr + offsetPara, d_cudaOddPsiPara.pitch, d_cudaOddPsiPara.pitch * dysize };
	PitchedPtr d_oddQPara = { (char*)d_cudaOddQPara.ptr + edgeOffsetPara, d_cudaOddQPara.pitch, d_cudaOddQPara.pitch * dysize };
#endif

#if COMPUTE_GROUND_STATE
	PitchedPtr d_HPsi = { (char*)d_cudaHPsi.ptr + offset, d_cudaHPsi.pitch, d_cudaHPsi.pitch * dysize };
#endif

	// find terms for laplacian
	Buffer<int3> d0;
	Buffer<int2> d1;
	Buffer<double> hodges;
#if HYPERBOLIC
	getLaplacian(hodges, d0, d1, sizeof(BlockPsis), d_evenPsiHyper.pitch, d_evenPsiHyper.slicePitch, sizeof(BlockEdges), d_evenQHyper.pitch, d_evenQHyper.slicePitch);
#endif
#if PARABOLIC
	getLaplacian(hodges, d0, d1, sizeof(BlockPsis), d_evenPsiPara.pitch, d_evenPsiPara.slicePitch, sizeof(BlockEdges), d_evenQPara.pitch, d_evenQPara.slicePitch);
#endif

	//std::cout << "lapsize = " << lapsize << ", lapfac = " << lapfac << ", lapfac0 = " << lapfac0 << std::endl;

	for (int i = 0; i < hodges.size(); ++i) hodges[i] = -0.5 * hodges[i] / (block_scale * block_scale);

	int3* d_d0 = allocDevice<int3>(d0.size());
	int2* d_d1 = allocDevice<int2>(d1.size());
	double* d_hodges = allocDevice<double>(hodges.size());

	// Initialize host memory
	size_t hostSize = dxsize * dysize * dzsize;
#if HYPERBOLIC
	BlockPsis* h_evenPsiHyper = allocHost<BlockPsis>(hostSize);
	BlockPsis* h_oddPsiHyper = allocHost<BlockPsis>(hostSize);
	BlockEdges* h_evenQHyper = allocHost<BlockEdges>(hostSize);
	BlockEdges* h_oddQHyper = allocHost<BlockEdges>(hostSize);
#endif
#if PARABOLIC
	BlockPsis* h_evenPsiPara = allocHost<BlockPsis>(hostSize);
	BlockPsis* h_oddPsiPara = allocHost<BlockPsis>(hostSize);
	BlockEdges* h_evenQPara = allocHost<BlockEdges>(hostSize);
	BlockEdges* h_oddQPara = allocHost<BlockEdges>(hostSize);
#endif
	double* h_density = allocHost<double>(bodies);

#if COMPUTE_GROUND_STATE
	// Initialize discrete field
	std::ifstream fs(GROUND_STATE_PSI_FILENAME, std::ios::binary | std::ios::in);
	bool continueFromEarlier = (fs.fail() == 0);
	if (continueFromEarlier)
	{
		std::cout << "Initialized ground state psi from file." << std::endl;

		fs.read((char*)&h_evenPsiHyper[0], hostSize * sizeof(BlockPsis));
		fs.close();

		std::ifstream fs_q(GROUND_STATE_Q_FILENAME, std::ios::binary | std::ios::in);
		if (fs.fail() == 0)
		{
			std::cout << "Initialized ground state q from file." << std::endl;

			fs.read((char*)&h_evenQHyper[0], hostSize * sizeof(BlockEdges));
			fs.close();
		}
		else
		{
			std::cout << "Failed to open the ground state q file." << std::endl;
		}
	}
	else
	{
		std::cout << "Initialized ground state with random noise." << std::endl;

		std::default_random_engine generator;
		std::normal_distribution<double> distribution(0.0, 1.0);
		for (uint k = 0; k < zsize; k++)
		{
			for (uint j = 0; j < ysize; j++)
			{
				for (uint i = 0; i < xsize; i++)
				{
					for (uint l = 0; l < bsize; l++)
					{
						const uint dstI = (k + 1) * dxsize * dysize + (j + 1) * dxsize + (i + 1);
						const double2 s1{ distribution(generator), distribution(generator) };
						const double2 s0{ distribution(generator), distribution(generator) };
						const double2 s_1{ distribution(generator), distribution(generator) };
						h_evenPsiHyper[dstI].values[l].s1 = s1;
						h_evenPsiHyper[dstI].values[l].s0 = s0;
						h_evenPsiHyper[dstI].values[l].s_1 = s_1;
					}
				}
			}
		}
	}

	bool loadGroundState = false;
	bool doForward = false;
#else
	bool loadGroundState = (t == 0);
	std::string psi_filename = loadGroundState ? GROUND_STATE_PSI_FILENAME : toString(t) + ".dat";
#if HYPERBOLIC
	loadFromFile(psi_filename, (char*)&h_oddPsiHyper[0], hostSize * sizeof(BlockPsis));
	std::string q_filename = loadGroundState ? GROUND_STATE_Q_FILENAME : toString(t) + ".dat";
	loadFromFile(q_filename, (char*)&h_oddQHyper[0], hostSize * sizeof(BlockEdges));
#endif
#if PARABOLIC
	loadFromFile(psi_filename, (char*)&h_oddPsiPara[0], hostSize * sizeof(BlockPsis));
#endif

	bool doForward = true;
#endif
#if HYPERBOLIC
	cudaPitchedPtr h_cudaEvenPsiHyper = copyHostToDevice3D(h_evenPsiHyper, d_cudaEvenPsiHyper, psiExtent);
	cudaPitchedPtr h_cudaOddPsiHyper = copyHostToDevice3D(h_oddPsiHyper, d_cudaOddPsiHyper, psiExtent);
	cudaPitchedPtr h_cudaEvenQHyper = copyHostToDevice3D(h_evenQHyper, d_cudaEvenQHyper, edgeExtent);
	cudaPitchedPtr h_cudaOddQHyper = copyHostToDevice3D(h_oddQHyper, d_cudaOddQHyper, edgeExtent);
#endif
#if PARABOLIC
	cudaPitchedPtr h_cudaEvenPsiPara = copyHostToDevice3D(h_evenPsiPara, d_cudaEvenPsiPara, psiExtent);
	cudaPitchedPtr h_cudaOddPsiPara = copyHostToDevice3D(h_oddPsiPara, d_cudaOddPsiPara, psiExtent);
	cudaPitchedPtr h_cudaEvenQPara = copyHostToDevice3D(h_evenQPara, d_cudaEvenQPara, edgeExtent);
	cudaPitchedPtr h_cudaOddQPara = copyHostToDevice3D(h_oddQPara, d_cudaOddQPara, edgeExtent);
#endif
	checkCudaErrors(cudaMemcpy(d_d0, &d0[0], d0.size() * sizeof(int3), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_d1, &d1[0], d1.size() * sizeof(int2), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_hodges, &hodges[0], hodges.size() * sizeof(double), cudaMemcpyHostToDevice));

	// Clear host memory after data has been copied to devices
	cudaDeviceSynchronize();
	d0.clear();
	d1.clear();
	hodges.clear();
#if !(SAVE_PICTURE)
	cudaFreeHost(h_evenPsiHyper);
	cudaFreeHost(h_oddPsiHyper);
#endif
#if HYPERBOLIC
	cudaMemcpy3DParms evenPsiBackParamsHyper = createDeviceToHostParams(d_cudaEvenPsiHyper, h_cudaEvenPsiHyper, psiExtent);
	cudaMemcpy3DParms oddPsiBackParamsHyper = createDeviceToHostParams(d_cudaOddPsiHyper, h_cudaOddPsiHyper, psiExtent);
	cudaMemcpy3DParms evenQBackParamsHyper = createDeviceToHostParams(d_cudaEvenQHyper, h_cudaEvenQHyper, edgeExtent);
	cudaMemcpy3DParms oddQBackParamsHyper = createDeviceToHostParams(d_cudaOddQHyper, h_cudaOddQHyper, edgeExtent);
#endif
#if PARABOLIC
	cudaMemcpy3DParms evenPsiBackParamsPara = createDeviceToHostParams(d_cudaEvenPsiPara, h_cudaEvenPsiPara, psiExtent);
	cudaMemcpy3DParms oddPsiBackParamsPara = createDeviceToHostParams(d_cudaOddPsiPara, h_cudaOddPsiPara, psiExtent);
	cudaMemcpy3DParms evenQBackParamsPara = createDeviceToHostParams(d_cudaEvenQPara, h_cudaEvenQPara, edgeExtent);
	cudaMemcpy3DParms oddQBackParamsPara = createDeviceToHostParams(d_cudaOddQPara, h_cudaOddQPara, edgeExtent);
#endif
	// Integrate in time
	uint3 dimensions = make_uint3(xsize, ysize, zsize);
	dim3 psiDimBlock(THREAD_BLOCK_X * VALUES_IN_BLOCK, THREAD_BLOCK_Y, THREAD_BLOCK_Z);
	dim3 edgeDimBlock(THREAD_BLOCK_X * EDGES_IN_BLOCK, THREAD_BLOCK_Y, THREAD_BLOCK_Z);
	dim3 dimGrid((xsize + THREAD_BLOCK_X - 1) / THREAD_BLOCK_X,
		(ysize + THREAD_BLOCK_Y - 1) / THREAD_BLOCK_Y,
		((zsize + THREAD_BLOCK_Z - 1) / THREAD_BLOCK_Z));

	Signal signal;
	MagFields Bs{};

	const double volume = block_scale * block_scale * block_scale * VOLUME;

#if COMPUTE_GROUND_STATE
	if (continueFromEarlier)
	{
		switch (initPhase)
		{
		case Phase::Polar:
			std::cout << "Transform ground state to polar phase" << std::endl;
			polarState << <dimGrid, psiDimBlock >> > (d_evenPsiHyper, dimensions);
			break;
		case Phase::Ferromagnetic:
			std::cout << "Transform ground state to ferromagnetic phase" << std::endl;
			ferromagneticState << <dimGrid, psiDimBlock >> > (d_evenPsiHyper, dimensions);
			break;
		default:
			break;
		}
		std::cout << "Total density: " << getDensity(dimGrid, psiDimBlock, d_density, d_evenPsiHyper, dimensions, bodies, volume) << std::endl;
	}
#endif

#if !COMPUTE_GROUND_STATE
	// Take one forward Euler step if starting from the ground state or time step changed
	if (doForward)
	{
		std::cout << "No even time step file found. Doing one forward step." << std::endl;

		signal = getSignal(t);
		Bs.Bq = BqScale * signal.Bq;
		Bs.Bb = BzScale * signal.Bb;
		Bs.BqQuad = BqQuadScale * signal.Bq;
		Bs.BbQuad = BzQuadScale * signal.Bb;
#if HYPERBOLIC
		forwardEuler << <dimGrid, psiDimBlock >> > (d_evenPsiHyper, d_oddPsiHyper, d_oddQHyper, d_d1, d_hodges, Bs, dimensions, block_scale, d_p0, c0, c2, dt, true);
		forwardEuler_q_hyper << <dimGrid, edgeDimBlock >> > (d_evenQHyper, d_oddQHyper, d_oddPsiHyper, d_d0, dimensions, dt_per_sigma);
#endif
#if PARABOLIC
		update_q_para << <dimGrid, edgeDimBlock >> > (d_oddQPara, d_oddQPara, d_oddPsiPara, d_d0, dimensions);
		forwardEuler << <dimGrid, psiDimBlock >> > (d_evenPsiPara, d_oddPsiPara, d_oddQPara, d_d1, d_hodges, Bs, dimensions, block_scale, d_p0, c0, c2, dt, false);
		update_q_para << <dimGrid, edgeDimBlock >> > (d_evenQPara, d_evenQPara, d_evenPsiPara, d_d0, dimensions);
#endif
	}
	else
#endif
	{
		std::cout << "Skipping the forward step." << std::endl;
	}

#if COMPUTE_GROUND_STATE
	std::string folder = "gs_dens_profiles_" + EXTRA_INFORMATION;
	std::string createResultsDirCommand = "mkdir " + folder;
	system(createResultsDirCommand.c_str());

	uint iter = 0;
	
	if (!continueFromEarlier)
	{
		normalize_h(dimGrid, psiDimBlock, d_density, d_evenPsiHyper, dimensions, bodies, volume);
		normalize_h(dimGrid, psiDimBlock, d_density, d_oddPsiHyper, dimensions, bodies, volume);
		itp_q_hyper << <dimGrid, edgeDimBlock >> > (d_evenQHyper, d_evenQHyper, d_evenPsiHyper, d_d0, dimensions, dt_per_sigma);
		itp_q_hyper << <dimGrid, edgeDimBlock >> > (d_oddQHyper, d_oddQHyper, d_oddPsiHyper, d_d0, dimensions, dt_per_sigma);
	}

	while (true)
	{
		if ((iter % 1000) == 0) std::cout << "Iteration " << iter << std::endl;
#if SAVE_PICTURE
		if ((iter % 1000) == 0)
		{
			checkCudaErrors(cudaMemcpy3D(&evenPsiBackParamsHyper));
			drawDensity("hyper", h_evenPsiHyper, dxsize, dysize, dzsize, iter, folder);
			std::cout << "Total density: " << getDensity(dimGrid, psiDimBlock, d_density, d_evenPsiHyper, dimensions, bodies, volume) << std::endl;

			// Compute energy/chemical potential
			innerProductReal << <dimGrid, psiDimBlock >> > (d_energy, d_evenPsiHyper, d_HPsi, dimensions);
			int prevStride = bodies;
			while (prevStride > 1)
			{
				int newStride = prevStride / 2;
				integrate << <dim3(std::ceil(newStride / 32.0), 1, 1), dim3(32, 1, 1) >> > (d_energy, newStride, ((newStride * 2) != prevStride), volume);
				prevStride = newStride;
			}
			double hEnergy = 0;
			checkCudaErrors(cudaMemcpy(&hEnergy, d_energy, sizeof(double), cudaMemcpyDeviceToHost));
			std::cout << "Energy: " << hEnergy << std::endl;
		}
#endif
		if (iter == 500000)
		{
			// Psi
			checkCudaErrors(cudaMemcpy3D(&evenPsiBackParamsHyper));
			std::ofstream fs_psi(GROUND_STATE_PSI_FILENAME, std::ios::binary | std::ios_base::trunc);
			if (fs_psi.fail() != 0) return 1;
			fs_psi.write((char*)&h_evenPsiHyper[0], hostSize * sizeof(BlockPsis));
			fs_psi.close();

			// Q
			checkCudaErrors(cudaMemcpy3D(&evenQBackParamsHyper));
			std::ofstream fs_q(GROUND_STATE_Q_FILENAME, std::ios::binary | std::ios_base::trunc);
			if (fs_q.fail() != 0) return 1;
			fs_q.write((char*)&h_evenQHyper[0], hostSize * sizeof(BlockEdges));
			fs_q.close();

			return 0;
		}

		// Take an imaginary time step
		itp_q_hyper << <dimGrid, edgeDimBlock >> > (d_oddQHyper, d_evenQHyper, d_evenPsiHyper, d_d0, dimensions, dt_per_sigma);
		itp_psi << <dimGrid, psiDimBlock >> > (d_HPsi, d_oddPsiHyper, d_evenPsiHyper, d_evenQHyper, d_d1, d_hodges, Bs, dimensions, block_scale, d_p0, c0, c2, dt);
		// Normalize
		normalize_h(dimGrid, psiDimBlock, d_density, d_oddPsiHyper, dimensions, bodies, volume);

		// Take an imaginary time step
		itp_q_hyper << <dimGrid, edgeDimBlock >> > (d_evenQHyper, d_oddQHyper, d_oddPsiHyper, d_d0, dimensions, dt_per_sigma);
		itp_psi << <dimGrid, psiDimBlock >> > (d_HPsi, d_evenPsiHyper, d_oddPsiHyper, d_oddQHyper, d_d1, d_hodges, Bs, dimensions, block_scale, d_p0, c0, c2, dt);
		// Normalize
		normalize_h(dimGrid, psiDimBlock, d_density, d_evenPsiHyper, dimensions, bodies, volume);

		iter++;
	}

#else
	int lastSaveTime = 0;

	std::string dens_folder = "dens_images_" + EXTRA_INFORMATION;

	std::string createResultsDirCommand = "mkdir " + dens_folder;
	system(createResultsDirCommand.c_str());

	while (t < END_TIME)
	{
#if SAVE_PICTURE
		// Copy back from device memory to host memory
#if HYPERBOLIC
		checkCudaErrors(cudaMemcpy3D(&oddPsiBackParamsHyper));
#endif
#if PARABOLIC
		checkCudaErrors(cudaMemcpy3D(&oddPsiBackParamsPara));
#endif

		// Measure wall clock time
		static auto prevTime = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::high_resolution_clock::now() - prevTime;
		//std::cout << "Simulation time: " << t << " ms. Real time from previous save: " << duration.count() * 1e-9 << " s." << std::endl;
		prevTime = std::chrono::high_resolution_clock::now();

#if !COMPUTE_ERROR
#if HYPERBOLIC
		double dens = getDensity(dimGrid, psiDimBlock, d_density, d_evenPsiHyper, dimensions, bodies, volume);
#endif
#if PARABOLIC
		double dens = getDensity(dimGrid, psiDimBlock, d_density, d_evenPsiPara, dimensions, bodies, volume);
#endif
		static double prevDens = dens;
		std::cout << "At " << t << " ms density is " << dens << std::endl;
		constexpr double MARGIN = 1.0; // 0.02;
		if (t > 0 && (abs(dens - prevDens) > MARGIN || isnan(dens)))
		{
			std::cout << "The final numerically stable time step size was " << dt - dt_increse << " ms!";
			return 1;
		}
		prevDens = dens;
#endif
#if HYPERBOLIC
		drawDensity("hyper", h_oddPsiHyper, dxsize, dysize, dzsize, t, dens_folder);
#endif
#if PARABOLIC
		drawDensity("para", h_oddPsiPara, dxsize, dysize, dzsize, t, dens_folder); 
#endif

#endif
		// integrate one iteration
		for (uint step = 0; step < IMAGE_SAVE_FREQUENCY; step++)
		{
			// update odd values (imaginary terms)
			t += dt / omega_r * 1e3; // [ms]
			signal = getSignal(t);
			Bs.Bq = BqScale * signal.Bq;
			Bs.Bb = BzScale * signal.Bb;
			Bs.BqQuad = BqQuadScale * signal.Bq;
			Bs.BbQuad = BzQuadScale * signal.Bb;

#if HYPERBOLIC
			update_psi << <dimGrid, psiDimBlock >> > (d_oddPsiHyper, d_evenPsiHyper, d_evenQHyper, d_d1, d_hodges, Bs, dimensions, block_scale, d_p0, c0, c2, dt, true);
			update_q_hyper << <dimGrid, edgeDimBlock >> > (d_oddQHyper, d_evenQHyper, d_evenPsiHyper, d_d0, dimensions, dt_per_sigma);
#endif
#if PARABOLIC
			update_psi << <dimGrid, psiDimBlock >> > (d_oddPsiPara, d_evenPsiPara, d_evenQPara, d_d1, d_hodges, Bs, dimensions, block_scale, d_p0, c0, c2, dt, false);
			update_q_para << <dimGrid, edgeDimBlock >> > (d_oddQPara, d_oddQPara, d_oddPsiPara, d_d0, dimensions);
#endif
			// update even values (real terms)
			t += dt / omega_r * 1e3; // [ms]
			signal = getSignal(t);
			Bs.Bq = BqScale * signal.Bq;
			Bs.Bb = BzScale * signal.Bb;
			Bs.BqQuad = BqQuadScale * signal.Bq;
			Bs.BbQuad = BzQuadScale * signal.Bb;
			
#if HYPERBOLIC
			update_psi << <dimGrid, psiDimBlock >> > (d_evenPsiHyper, d_oddPsiHyper, d_oddQHyper, d_d1, d_hodges, Bs, dimensions, block_scale, d_p0, c0, c2, dt, true);
			update_q_hyper << <dimGrid, edgeDimBlock >> > (d_evenQHyper, d_oddQHyper, d_oddPsiHyper, d_d0, dimensions, dt_per_sigma);
#endif
#if PARABOLIC
			update_psi << <dimGrid, psiDimBlock >> > (d_evenPsiPara, d_oddPsiPara, d_oddQPara, d_d1, d_hodges, Bs, dimensions, block_scale, d_p0, c0, c2, dt, false);
			update_q_para << <dimGrid, edgeDimBlock >> > (d_evenQPara, d_evenQPara, d_evenPsiPara, d_d0, dimensions);
#endif
		}
#if COMPUTE_ERROR
		// Compute error
		//innerProduct << <dimGrid, psiDimBlock >> > (d_error, d_evenPsiPara, d_evenPsiHyper, dimensions);
		weightedDiff << <dimGrid, psiDimBlock >> > (d_error, d_evenPsiPara, d_evenPsiHyper, dimensions);
		int prevStride = bodies;
		while (prevStride > 1)
		{
			int newStride = prevStride / 2;
			integrate << <dim3(std::ceil(newStride / 32.0), 1, 1), dim3(32, 1, 1) >> > (d_error, newStride, ((newStride * 2) != prevStride), volume);
			prevStride = newStride;
		}
		//double2 hError = { 0 };
		double hError = { 0 };
		checkCudaErrors(cudaMemcpy(&hError, d_error, sizeof(double2), cudaMemcpyDeviceToHost));
		//std::cout << getDensity(dimGrid, psiDimBlock, d_density, d_evenPsiPara, dimensions, bodies, volume) - sqrt((conj(hError) * hError).x) << ", ";
		std::cout << hError << ", ";
		//std::cout << getDensity(dimGrid, psiDimBlock, d_density, d_evenPsiPara, dimensions, bodies, volume) - hError.x << ", ";
#endif

#if COMPUTE_GROUND_STATE
		// Copy back from device memory to host memory
		checkCudaErrors(cudaMemcpy3D(&oddPsiBackParamsHyper));

		temp = h_oddPsiHyper[centerIdx].values[5].s0;
		double endPhase = atan2(temp.y, temp.x);
		double phaseDiff = endPhase - startPhase;
		std::cout << "Energy was " << phaseDiff / phaseTime << std::endl;
#endif
#if SAVE_STATES
		// Copy back from device memory to host memory
		checkCudaErrors(cudaMemcpy3D(&oddPsiBackParamsHyper));



		if (((int(t) % STATE_SAVE_INTERVAL) == 0) && (int(t) != lastSaveTime))
		{
			std::ofstream oddFs(SAVE_FILE_PREFIX + toString(t) + ".dat", std::ios::binary | std::ios_base::trunc);
			if (oddFs.fail() != 0) return 1;
			oddFs.write((char*)&h_oddPsiHyper[0], hostSize * sizeof(BlockPsis));
			oddFs.close();

			checkCudaErrors(cudaMemcpy3D(&evenPsiBackParamsHyper));
			std::ofstream evenFs(SAVE_FILE_PREFIX + "even_" + toString(t) + ".dat", std::ios::binary | std::ios_base::trunc);
			if (evenFs.fail() != 0) return 1;
			evenFs.write((char*)&h_evenPsiHyper[0], hostSize * sizeof(BlockPsis));
			evenFs.close();

			std::cout << "Saved the state!" << std::endl;

			lastSaveTime = int(t);
		}
#endif
	}
#endif
#if HYPERBOLIC
	checkCudaErrors(cudaFree(d_cudaEvenPsiHyper.ptr));
	checkCudaErrors(cudaFree(d_cudaEvenQHyper.ptr));
	checkCudaErrors(cudaFree(d_cudaOddPsiHyper.ptr));
	checkCudaErrors(cudaFree(d_cudaOddQHyper.ptr));

	checkCudaErrors(cudaFreeHost(h_evenPsiHyper));
	checkCudaErrors(cudaFreeHost(h_oddPsiHyper));
	checkCudaErrors(cudaFreeHost(h_evenQHyper));
	checkCudaErrors(cudaFreeHost(h_oddQHyper));
#endif
#if PARABOLIC
	checkCudaErrors(cudaFree(d_cudaEvenPsiPara.ptr));
	checkCudaErrors(cudaFree(d_cudaEvenQPara.ptr));
	checkCudaErrors(cudaFree(d_cudaOddPsiPara.ptr));
	checkCudaErrors(cudaFree(d_cudaOddQPara.ptr));

	checkCudaErrors(cudaFreeHost(h_evenPsiPara));
	checkCudaErrors(cudaFreeHost(h_oddPsiPara));
	checkCudaErrors(cudaFreeHost(h_evenQPara));
	checkCudaErrors(cudaFreeHost(h_oddQPara));
#endif
#if COMPUTE_ERROR
	checkCudaErrors(cudaFree(d_error));
#endif
#if COMPUTE_GROUND_STATE
	checkCudaErrors(cudaFree(d_cudaHPsi.ptr));
	checkCudaErrors(cudaFree(d_energy));
#endif
	//checkCudaErrors(cudaFree(d_spinNorm));
	checkCudaErrors(cudaFree(d_density));
	checkCudaErrors(cudaFree(d_d0));
	checkCudaErrors(cudaFree(d_d1));
	checkCudaErrors(cudaFree(d_hodges));

	checkCudaErrors(cudaFreeHost(h_density));

	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to launch kernels (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	return 0;
}

void readConfFile()
{
	std::ifstream file;
	file.open("conf.conf", std::ios::in);
	if (file.is_open())
	{
		std::string line;
		while (std::getline(file, line))
		{
			if (size_t pos = line.find("t0") != std::string::npos)
			{
				t = std::stod(line.substr(pos + 2));
			}
			else if (size_t pos = line.find("end") != std::string::npos)
			{
				END_TIME = std::stod(line.substr(pos + 3));
			}
			else if (size_t pos = line.find("dt") != std::string::npos)
			{
				dt = std::stod(line.substr(pos + 2));
				IMAGE_SAVE_FREQUENCY = uint(IMAGE_SAVE_INTERVAL * 0.5 / 1e3 * omega_r / dt) + 1;
				dt_per_sigma = dt / sigma;
			}
			else if (size_t pos = line.find("sigma") != std::string::npos)
			{
				sigma = std::stod(line.substr(pos + 5));
				dt_per_sigma = dt / sigma;
			}
		}
	}
}

int main(int argc, char** argv)
{
	readConfFile();

	const double blockScale = DOMAIN_SIZE_X / REPLICABLE_STRUCTURE_COUNT_X / BLOCK_WIDTH_X;

	std::cout << "Start simulating from t = " << t << " ms." << std::endl;
	std::cout << "The simulation will end at " << END_TIME << " ms." << std::endl;
	std::cout << "Block scale = " << blockScale << std::endl;
	std::cout << "Dual edge length = " << DUAL_EDGE_LENGTH * blockScale << std::endl;
	std::cout << "Relativistic sigma = " << sigma << std::endl;

	// integrate in time using DEC
	auto domainMin = Vector3(-DOMAIN_SIZE_X * 0.5, -DOMAIN_SIZE_Y * 0.5, -DOMAIN_SIZE_Z * 0.5);
	auto domainMax = Vector3(DOMAIN_SIZE_X * 0.5, DOMAIN_SIZE_Y * 0.5, DOMAIN_SIZE_Z * 0.5);

	while (!integrateInTime(blockScale, domainMin, domainMax))
	{
		std::cout << "Time step was: " << dt << std::endl;
		dt += dt_increse;
		dt_per_sigma = dt / sigma;
		IMAGE_SAVE_FREQUENCY = uint(IMAGE_SAVE_INTERVAL * 0.5 / 1e3 * omega_r / dt) + 1;

		t = 0;
	}

	return 0;
}
