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
#include <iomanip>

#include "mesh.h"

#define COMPUTE_GROUND_STATE 0

#define HYPERBOLIC 1
#define PARABOLIC 0
#define ANALYTIC 0
#define COMPUTE_STABLE_DT 0
#define COMPUTE_ERROR ((HYPERBOLIC && PARABOLIC) || (ANALYTIC && PARABOLIC) || (HYPERBOLIC && ANALYTIC))
#define COMPUTE_COM 0

#define SAVE_STATES 0
#define SAVE_PICTURE 1

#define THREAD_BLOCK_X 16
#define THREAD_BLOCK_Y 2
#define THREAD_BLOCK_Z 1

constexpr myFloat F = 1; // Hyperfine spin

constexpr myFloat DOMAIN_SIZE_X = 20.0; //24.0;
constexpr myFloat DOMAIN_SIZE_Y = 20.0; //24.0;
constexpr myFloat DOMAIN_SIZE_Z = 20.0; //24.0;

myFloat REPLICABLE_STRUCTURE_COUNT_X = 112.0; // 58.0 + 9 * 6.0;
//constexpr myFloat REPLICABLE_STRUCTURE_COUNT_Y = 112.0;
//constexpr myFloat REPLICABLE_STRUCTURE_COUNT_Z = 112.0;

constexpr myFloat N = 2e5; // Number of atoms in the condensate

constexpr myFloat trapFreq_r = 126; // Cloud oscillates at 125.91 Hz in sims
constexpr myFloat trapFreq_z = 166;

constexpr myFloat omega_r = trapFreq_r * 2 * PI;
constexpr myFloat omega_z = trapFreq_z * 2 * PI;
constexpr myFloat lambda_x = 1.0;
constexpr myFloat lambda_y = 1.0;
constexpr myFloat lambda_z = omega_z / omega_r;

constexpr myFloat a_bohr = 5.2917721092e-11; //[m] Bohr radius
constexpr myFloat a_0 = 101.8;
constexpr myFloat a_2 = 100.4;

constexpr myFloat atomMass = 1.44316060e-25;
constexpr myFloat hbar = 1.05457148e-34; // [m^2 kg / s]
const myFloat a_r = sqrt(hbar / (atomMass * omega_r)); //[m]

const myFloat c0 = 4 * PI * N * (a_0 + 2 * a_2) * a_bohr / (3 * a_r);
const myFloat c2 = 4 * PI * N * (a_2 - a_0) * a_bohr / (3 * a_r);

constexpr myFloat muB = 9.27400968e-24; // [m^2 kg / s^2 T^-1] Bohr magneton

const myFloat BqScale = -(0.5 * muB / (hbar * omega_r) * a_r) / 100.; // [cm/Gauss]
constexpr myFloat BzScale = -(0.5 * muB / (hbar * omega_r)) / 10000.; // [1/Gauss]

constexpr myFloat A_hfs = 3.41734130545215;
const myFloat BqQuadScale = 100 * a_r * sqrt(0.25 * 1000 * (1.399624624 * 1.399624624) / (trapFreq_r * 2 * A_hfs)); //[cm/Gauss]
const myFloat BzQuadScale = sqrt(0.25 * 1000 * (1.399624624 * 1.399624624) / (trapFreq_r * 2 * A_hfs)); //[1/Gauss]  \sqrt{g_q}

constexpr myFloat SQRT_2 = 1.41421356237309;
constexpr myFloat INV_SQRT_2 = 0.70710678118655;

//myFloat dt = 3.9e-4; // For parabolic eq and 112^3 domain
//myFloat dt = 6.9e-4; // For hyperbolic eq and 112^3 domain
//myFloat dt = 1e-4; // Default
//myFloat dt = 1.2e-3;
myFloat dt = 5e-5;
#if HYPERBOLIC
//myFloat dt = 1.12e-3;
#elif PARABOLIC
//myFloat dt = 3.9e-4;
#endif
myFloat dt_decrese = 1e-4;
myFloat dt_increse = 1e-5;

const myFloat IMAGE_SAVE_INTERVAL = 0.02; // ms
uint IMAGE_SAVE_FREQUENCY = uint(IMAGE_SAVE_INTERVAL * 0.5 / 1e3 * omega_r / dt) + 1;

const uint STATE_SAVE_INTERVAL = 10.0; // ms

myFloat t = 0; // Start time in ms
#if COMPUTE_COM
myFloat END_TIME = 10.0; // End time in ms
#else
myFloat END_TIME = 0.55; // End time in ms
#endif

#if COMPUTE_GROUND_STATE
myFloat tau = 0.1; // 0.01; // Coefficient for the relativistic term (zero for non-relativistic)
#else
myFloat tau = 0.005; // 0.01; // Coefficient for the relativistic term
#endif
myFloat dt_per_tau = dt / tau;
static int dtIncreaseCount = 0;

//constexpr myFloat E = 126.7;
//constexpr myFloat E = 127.346; // Computed with the hyperbolic ITP
constexpr myFloat E = 127.333; // Computed with the hyperbolic ITP
//constexpr myFloat E = 127.314; // Parabolic
//constexpr myFloat E = 127.295; // Hyperbolic

/// Hyperbolic
/// Energy was -127.333
/// Parabolic
/// 
/// /// Parabolic
/// Energy was -127.297
/// Energy was - 127.302
/// Energy was - 127.294
/// Parabolic

enum class Phase
{
	Polar = 0,
	Ferromagnetic,
	None
};
//constexpr Phase initPhase = Phase::None;
constexpr Phase initPhase = Phase::Polar;
//constexpr Phase initPhase = Phase::Ferromagnetic;

std::string toStringShort(const myFloat value)
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

#include "hyper_kernels.h"
#include "para_kernels.h"
#include "common_kernels.h"

void normalize_h(dim3 dimGrid, dim3 dimBlock, myFloat* densityPtr, PitchedPtr psi, uint3 dimensions, size_t bodies, myFloat volume)
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

myFloat getDensity(dim3 dimGrid, dim3 dimBlock, myFloat* densityPtr, PitchedPtr psi, uint3 dimensions, size_t bodies, myFloat volume)
{
	density << <dimGrid, dimBlock >> > (densityPtr, psi, dimensions);
	int prevStride = bodies;
	while (prevStride > 1)
	{
		int newStride = prevStride / 2;
		integrate << <dim3(std::ceil(newStride / 32.0), 1, 1), dim3(32, 1, 1) >> > (densityPtr, newStride, ((newStride * 2) != prevStride), volume);
		prevStride = newStride;
	}
	myFloat hDensity = 0;
	checkCudaErrors(cudaMemcpy(&hDensity, densityPtr, sizeof(myFloat), cudaMemcpyDeviceToHost));

	return hDensity;
}

myFloat3 centerOfMass(dim3 dimGrid, dim3 dimBlock, myFloat3* comPtr, PitchedPtr psi, uint3 dimensions, size_t bodies, const myFloat volume, const myFloat block_scale, const myFloat3 p0)
{
	com << <dimGrid, dimBlock >> > (comPtr, psi, dimensions, block_scale, p0);
	int prevStride = bodies;
	while (prevStride > 1)
	{
		int newStride = prevStride / 2;
		integrate << <dim3(std::ceil(newStride / 32.0), 1, 1), dim3(32, 1, 1) >> > (comPtr, newStride, ((newStride * 2) != prevStride), volume);
		prevStride = newStride;
	}
	myFloat3 hCom = { 0, 0, 0 };
	checkCudaErrors(cudaMemcpy(&hCom, comPtr, sizeof(myFloat), cudaMemcpyDeviceToHost));

	return hCom;
}

struct SpinMagDens
{
	myFloat spin;
	myFloat3 magnetization;
	myFloat density;
};

SpinMagDens integrateSpinAndDensity(dim3 dimGrid, dim3 dimBlock, myFloat* spinNormPtr, myFloat3* localAvgSpinPtr, myFloat* densityPtr, size_t bodies, myFloat volume)
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
			integrate << <dim3(std::ceil(newStride / 32.0), 1, 1), dim3(32, 1, 1) >> > (localAvgSpinPtr, newStride, ((newStride * 2) != prevStride), volume);
		}
		integrate << <dim3(std::ceil(newStride / 32.0), 1, 1), dim3(32, 1, 1) >> > (spinNormPtr, newStride, ((newStride * 2) != prevStride), volume);
		integrate << <dim3(std::ceil(newStride / 32.0), 1, 1), dim3(32, 1, 1) >> > (densityPtr, newStride, ((newStride * 2) != prevStride), volume);
		prevStride = newStride;
	}
	myFloat3 hMagnetization = { 0, 0, 0 };
	checkCudaErrors(cudaMemcpy(&hMagnetization, localAvgSpinPtr, sizeof(myFloat3), cudaMemcpyDeviceToHost));

	myFloat hSpinNorm = 0;
	checkCudaErrors(cudaMemcpy(&hSpinNorm, spinNormPtr, sizeof(myFloat), cudaMemcpyDeviceToHost));

	myFloat hDensity = 0;
	checkCudaErrors(cudaMemcpy(&hDensity, densityPtr, sizeof(myFloat), cudaMemcpyDeviceToHost));

	return { hSpinNorm, hMagnetization, hDensity };
}

myFloat getMaxHamilton(dim3 dimGrid, dim3 dimBlock, myFloat* maxHamlPtr, PitchedPtr psi, MagFields Bs, uint3 dimensions, size_t bodies, myFloat block_scale, myFloat3 p0)
{
	maxHamilton << <dimGrid, dimBlock >> > (maxHamlPtr, psi, Bs, dimensions, block_scale, p0, c0, c2);
	int prevStride = bodies;
	while (prevStride > 1)
	{
		int newStride = prevStride / 2;
		reduceMax << <dim3(std::ceil(newStride / 32.0), 1, 1), dim3(32, 1, 1) >> > (maxHamlPtr, newStride, ((newStride * 2) != prevStride));
		prevStride = newStride;
	}

	myFloat maxHaml = 0;
	checkCudaErrors(cudaMemcpy(&maxHaml, maxHamlPtr, sizeof(myFloat), cudaMemcpyDeviceToHost));

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
		//std::cout << "Loading data from " << filename << "..." << std::endl;
	}
	psi_fs.read(dst, size);
	psi_fs.close();
}

uint integrateInTime(const myFloat block_scale, const Vector3& minp, const Vector3& maxp)
{
	const std::string EXTRA_INFORMATION = toStringShort(DOMAIN_SIZE_X) + "_" + toStringShort(REPLICABLE_STRUCTURE_COUNT_X);
	const std::string GROUND_STATE_PSI_FILENAME = "ground_state_psi_" + EXTRA_INFORMATION + "_" + PRECISION + ".dat";
	const std::string GROUND_STATE_Q_FILENAME = "ground_state_q_" + EXTRA_INFORMATION + "_" + PRECISION + ".dat";

	// find dimensions
	const Vector3 domain = maxp - minp;
	const uint xsize = uint(domain.x / (block_scale * BLOCK_WIDTH.x)); // + 1;
	const uint ysize = uint(domain.y / (block_scale * BLOCK_WIDTH.y)); // + 1;
	const uint zsize = uint(domain.z / (block_scale * BLOCK_WIDTH.z)); // + 1;
	const Vector3 p0 = 0.5 * (minp + maxp - block_scale * Vector3(BLOCK_WIDTH.x * xsize, BLOCK_WIDTH.y * ysize, BLOCK_WIDTH.z * zsize));
#if COMPUTE_COM
	const myFloat3 d_p0 = { p0.x + 0.14 * maxp.x, p0.y, p0.z };
#else
	const myFloat3 d_p0 = { p0.x, p0.y, p0.z };
#endif

	// compute discrete dimensions
	const uint bsize = VALUES_IN_BLOCK; // bpos.size(); // number of values inside a block

	//std::cout << "Dual 0-cells in a replicable structure: " << bsize << std::endl;
	//std::cout << "Replicable structure instances in x: " << xsize << ", y: " << ysize << ", z: " << zsize << std::endl;
	uint64_t bodies = xsize * ysize * zsize * bsize;
	//std::cout << "Dual 0-cells in total: " << bodies << std::endl;

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
#if ANALYTIC
	cudaPitchedPtr d_cudaGroundPsi = allocDevice3D(psiExtent);
	cudaPitchedPtr d_cudaAnalyticPsi = allocDevice3D(psiExtent);
#endif

#if COMPUTE_ERROR || ANALYTIC
	myFloat2* d_error = allocDevice<myFloat2>(bodies);
	//myFloat* d_error = allocDevice<myFloat>(bodies);
#endif

#if COMPUTE_GROUND_STATE
	// For computing the energy/chemical potential
	cudaPitchedPtr d_cudaHPsi = allocDevice3D(edgeExtent);
#endif

	//myFloat* d_spinNorm = allocDevice<myFloat>(bodies);
	myFloat* d_density = allocDevice<myFloat>(bodies);
	myFloat3* d_com = allocDevice<myFloat3>(bodies);
#if COMPUTE_GROUND_STATE
	myFloat* d_energy = allocDevice<myFloat>(bodies);
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
#if ANALYTIC
	size_t offsetAnal = d_cudaAnalyticPsi.pitch * dysize + d_cudaAnalyticPsi.pitch + sizeof(BlockPsis);
	PitchedPtr d_groundPsi = { (char*)d_cudaGroundPsi.ptr + offsetAnal, d_cudaGroundPsi.pitch, d_cudaGroundPsi.pitch * dysize };
	PitchedPtr d_analyticPsi = { (char*)d_cudaAnalyticPsi.ptr + offsetAnal, d_cudaAnalyticPsi.pitch, d_cudaAnalyticPsi.pitch * dysize };
#endif

#if COMPUTE_GROUND_STATE
	PitchedPtr d_HPsi = { (char*)d_cudaHPsi.ptr + offsetHyper, d_cudaHPsi.pitch, d_cudaHPsi.pitch * dysize };
#endif

	// find terms for laplacian
	Buffer<int3> d0;
	Buffer<int2> d1;
	Buffer<myFloat> hodges;
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
	myFloat* d_hodges = allocDevice<myFloat>(hodges.size());

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
#if ANALYTIC
	BlockPsis* h_analyticPsi = allocHost<BlockPsis>(hostSize);
#endif
	myFloat* h_density = allocHost<myFloat>(bodies);

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
		std::normal_distribution<myFloat> distribution(0.0, 1.0);
		for (uint k = 0; k < zsize; k++)
		{
			for (uint j = 0; j < ysize; j++)
			{
				for (uint i = 0; i < xsize; i++)
				{
					for (uint l = 0; l < bsize; l++)
					{
						const uint dstI = (k + 1) * dxsize * dysize + (j + 1) * dxsize + (i + 1);
						const myFloat2 s1{ distribution(generator), distribution(generator) };
						const myFloat2 s0{ distribution(generator), distribution(generator) };
						const myFloat2 s_1{ distribution(generator), distribution(generator) };
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
	//std::string q_filename = loadGroundState ? GROUND_STATE_Q_FILENAME : toString(t) + ".dat";
	//loadFromFile(q_filename, (char*)&h_oddQHyper[0], hostSize * sizeof(BlockEdges));
#endif
#if PARABOLIC
	loadFromFile(psi_filename, (char*)&h_oddPsiPara[0], hostSize * sizeof(BlockPsis));
#endif
#if ANALYTIC
	loadFromFile(psi_filename, (char*)&h_analyticPsi[0], hostSize * sizeof(BlockPsis));
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
#if ANALYTIC
	cudaPitchedPtr h_cudaAnalyticPsi = copyHostToDevice3D(h_analyticPsi, d_cudaGroundPsi, psiExtent);
#endif

	checkCudaErrors(cudaMemcpy(d_d0, &d0[0], d0.size() * sizeof(int3), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_d1, &d1[0], d1.size() * sizeof(int2), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_hodges, &hodges[0], hodges.size() * sizeof(myFloat), cudaMemcpyHostToDevice));

	// Clear host memory after data has been copied to devices
	cudaDeviceSynchronize();
	d0.clear();
	d1.clear();
	hodges.clear();

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
#if ANALYTIC
	cudaMemcpy3DParms analyticPsiBackParams = createDeviceToHostParams(d_cudaAnalyticPsi, h_cudaAnalyticPsi, psiExtent);
#endif
	// Integrate in time
	uint3 dimensions = make_uint3(xsize, ysize, zsize);
	dim3 psiDimBlock(THREAD_BLOCK_X * VALUES_IN_BLOCK, THREAD_BLOCK_Y, THREAD_BLOCK_Z);
	dim3 edgeDimBlock(THREAD_BLOCK_X * EDGES_IN_BLOCK, THREAD_BLOCK_Y, THREAD_BLOCK_Z);
	dim3 dimGrid((xsize + THREAD_BLOCK_X - 1) / THREAD_BLOCK_X,
		         (ysize + THREAD_BLOCK_Y - 1) / THREAD_BLOCK_Y,
		         (zsize + THREAD_BLOCK_Z - 1) / THREAD_BLOCK_Z);

	Signal signal;
	MagFields Bs{};

	const myFloat volume = block_scale * block_scale * block_scale * VOLUME;

#if !COMPUTE_GROUND_STATE
#if HYPERBOLIC
	switch (initPhase)
	{
	case Phase::Polar:
		//std::cout << "Transform ground state to polar phase" << std::endl;
		polarState << <dimGrid, psiDimBlock >> > (d_oddPsiHyper, dimensions);
		break;
	case Phase::Ferromagnetic:
		//std::cout << "Transform ground state to ferromagnetic phase" << std::endl;
		ferromagneticState << <dimGrid, psiDimBlock >> > (d_oddPsiHyper, dimensions);
		break;
	default:
		break;
	}
#endif
#if PARABOLIC
	switch (initPhase)
	{
	case Phase::Polar:
		//std::cout << "Transform ground state to polar phase" << std::endl;
		polarState << <dimGrid, psiDimBlock >> > (d_oddPsiPara, dimensions);
		break;
	case Phase::Ferromagnetic:
		//std::cout << "Transform ground state to ferromagnetic phase" << std::endl;
		ferromagneticState << <dimGrid, psiDimBlock >> > (d_oddPsiPara, dimensions);
		break;
	default:
		break;
	}
#endif
#if ANALYTIC
	switch (initPhase)
	{
	case Phase::Polar:
		//std::cout << "Transform ground state to polar phase" << std::endl;
		polarState << <dimGrid, psiDimBlock >> > (d_groundPsi, dimensions);
		break;
	case Phase::Ferromagnetic:
		//std::cout << "Transform ground state to ferromagnetic phase" << std::endl;
		ferromagneticState << <dimGrid, psiDimBlock >> > (d_groundPsi, dimensions);
		break;
	default:
		break;
	}
#endif

	// Take one forward Euler step if starting from the ground state or time step changed
	if (doForward)
	{
		//std::cout << "No even time step file found. Doing one forward step." << std::endl;

		signal = getSignal(t);
		Bs.Bq = BqScale * signal.Bq;
		Bs.Bb = BzScale * signal.Bb;
		Bs.BqQuad = BqQuadScale * signal.Bq;
		Bs.BbQuad = BzQuadScale * signal.Bb;
#if HYPERBOLIC
		update_q_para << <dimGrid, edgeDimBlock >> > (d_oddQHyper, d_oddPsiHyper, d_d0, dimensions);
		forwardEuler << <dimGrid, psiDimBlock >> > (d_evenPsiHyper, d_oddPsiHyper, d_oddQHyper, d_d1, d_hodges, Bs, dimensions, block_scale, d_p0, c0, c2, dt, true, tau);
		update_q_para << <dimGrid, edgeDimBlock >> > (d_evenQHyper, d_evenPsiHyper, d_d0, dimensions);
		//forwardEuler_q_hyper << <dimGrid, edgeDimBlock >> > (d_evenQHyper, d_oddQHyper, d_oddPsiHyper, d_d0, dimensions, dt_per_tau);
#endif
#if PARABOLIC
		update_q_para << <dimGrid, edgeDimBlock >> > (d_oddQPara, d_oddPsiPara, d_d0, dimensions);
		forwardEuler << <dimGrid, psiDimBlock >> > (d_evenPsiPara, d_oddPsiPara, d_oddQPara, d_d1, d_hodges, Bs, dimensions, block_scale, d_p0, c0, c2, dt, false, tau);
		update_q_para << <dimGrid, edgeDimBlock >> > (d_evenQPara, d_evenPsiPara, d_d0, dimensions);
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
		itp_q_hyper << <dimGrid, edgeDimBlock >> > (d_evenQHyper, d_evenQHyper, d_evenPsiHyper, d_d0, dimensions, dt_per_tau);
		itp_q_hyper << <dimGrid, edgeDimBlock >> > (d_oddQHyper, d_oddQHyper, d_oddPsiHyper, d_d0, dimensions, dt_per_tau);
	}

	while (true)
	{
		constexpr int ITERS_PER_IMAGE = 10000;
		if ((iter % ITERS_PER_IMAGE) == 0) std::cout << "Iteration " << iter << std::endl;
#if SAVE_PICTURE
		if ((iter % ITERS_PER_IMAGE) == 0)
		{
			checkCudaErrors(cudaMemcpy3D(&evenPsiBackParamsHyper));
			drawIandR(folder, h_evenPsiHyper, dxsize, dysize, dzsize, iter, Bs, d_p0, block_scale);
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
			myFloat hEnergy = 0;
			checkCudaErrors(cudaMemcpy(&hEnergy, d_energy, sizeof(myFloat), cudaMemcpyDeviceToHost));
			std::setprecision(15);
			std::cout << "Energy: " << hEnergy << std::endl;
		}
#endif
		if (iter == 100000)
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
		itp_q_hyper << <dimGrid, edgeDimBlock >> > (d_oddQHyper, d_evenQHyper, d_evenPsiHyper, d_d0, dimensions, dt_per_tau);
		itp_psi << <dimGrid, psiDimBlock >> > (d_HPsi, d_oddPsiHyper, d_evenPsiHyper, d_evenQHyper, d_d1, d_hodges, Bs, dimensions, block_scale, d_p0, c0, c2, dt);
		// Normalize
		normalize_h(dimGrid, psiDimBlock, d_density, d_oddPsiHyper, dimensions, bodies, volume);

		// Take an imaginary time step
		itp_q_hyper << <dimGrid, edgeDimBlock >> > (d_evenQHyper, d_oddQHyper, d_oddPsiHyper, d_d0, dimensions, dt_per_tau);
		itp_psi << <dimGrid, psiDimBlock >> > (d_HPsi, d_evenPsiHyper, d_oddPsiHyper, d_oddQHyper, d_d1, d_hodges, Bs, dimensions, block_scale, d_p0, c0, c2, dt);
		// Normalize
		normalize_h(dimGrid, psiDimBlock, d_density, d_evenPsiHyper, dimensions, bodies, volume);

		iter++;
	}

#else
	int lastSaveTime = 0;

	std::string dens_folder = "dens_images_" + EXTRA_INFORMATION + "_" + getBasisString() + "_" + phaseToString(initPhase);
	std::string vtks_folder = "vtks_" + EXTRA_INFORMATION;
#if !COMPUTE_COM
	std::string createResultsDirCommand = "mkdir " + dens_folder;
	std::string createVtksDirCommand = "mkdir " + vtks_folder;
	system(createResultsDirCommand.c_str());
	system(createVtksDirCommand.c_str());
#endif
#if 1// ANALYTIC
	myFloat phaseTime = 0;
#endif
	int iterCount = 0;
#if COMPUTE_ERROR
	std::cout << "e_F1 = [";
#endif

#if ANALYTIC
	// Compute error
	myFloat2 phaseShift = myFloat2{ cos(-phaseTime * E), sin(-phaseTime * E) };
	analyticStep << <dimGrid, psiDimBlock >> > (d_analyticPsi, d_groundPsi, dimensions, phaseShift);

#if HYPERBOLIC
	weightedDiff << <dimGrid, psiDimBlock >> > (d_error, d_analyticPsi, d_oddPsiHyper, dimensions);
#elif PARABOLIC
	weightedDiff << <dimGrid, psiDimBlock >> > (d_error, d_analyticPsi, d_oddPsiPara, dimensions);
#endif
	int prevStride = bodies;
	while (prevStride > 1)
	{
		int newStride = prevStride / 2;
		integrate << <dim3(std::ceil(newStride / 32.0), 1, 1), dim3(32, 1, 1) >> > (d_error, newStride, ((newStride * 2) != prevStride), volume);
		prevStride = newStride;
	}
	myFloat2 hError = { 0 };
	//myFloat hError = { 0 };
	checkCudaErrors(cudaMemcpy(&hError, d_error, sizeof(myFloat2), cudaMemcpyDeviceToHost));
	//std::cout << getDensity(dimGrid, psiDimBlock, d_density, d_evenPsiPara, dimensions, bodies, volume) - sqrt((conj(hError) * hError).x) << ", ";
	std::cout << hError.x << ", " << hError.y << "; ";
	//std::cout << getDensity(dimGrid, psiDimBlock, d_density, d_evenPsiPara, dimensions, bodies, volume) - hError.x << ", ";
#endif

	// Measure wall clock time
	static auto prevTime = std::chrono::high_resolution_clock::now();

#if COMPUTE_STABLE_DT
	while (iterCount < 10000)
#else
	while (t < END_TIME)
#endif
	{
		// Measure wall clock time
		//static auto prevTime = std::chrono::high_resolution_clock::now();
		//auto duration = std::chrono::high_resolution_clock::now() - prevTime;
		////std::cout << "Simulation time: " << t << " ms. Real time from previous save: " << duration.count() * 1e-9 << " s." << std::endl;
		//prevTime = std::chrono::high_resolution_clock::now();

#if SAVE_PICTURE
#if HYPERBOLIC
		checkCudaErrors(cudaMemcpy3D(&oddPsiBackParamsHyper));
		drawDensity("hyper", h_oddPsiHyper, dxsize, dysize, dzsize, t, dens_folder);
		savePreImageSpinor(vtks_folder, h_oddPsiHyper, bsize, dxsize, dysize, dzsize, block_scale, d_p0, t);
#endif
#if PARABOLIC
		checkCudaErrors(cudaMemcpy3D(&oddPsiBackParamsPara));
		drawDensityRI("para", h_oddPsiPara, dxsize, dysize, dzsize, t, dens_folder);
		savePreImageSpinor(vtks_folder, h_oddPsiPara, bsize, dxsize, dysize, dzsize, block_scale, d_p0, t);
#endif
#if ANALYTIC
		checkCudaErrors(cudaMemcpy3D(&analyticPsiBackParams));
		drawDensityRI("analytic_", h_analyticPsi, dxsize, dysize, dzsize, t, dens_folder);
#endif
#endif


		// For checking the numerical stability
#if COMPUTE_STABLE_DT // Disable / enable numerical stability measurement
#if !COMPUTE_ERROR
#if HYPERBOLIC
		myFloat dens = getDensity(dimGrid, psiDimBlock, d_density, d_evenPsiHyper, dimensions, bodies, volume);
#endif
#if PARABOLIC
		myFloat dens = getDensity(dimGrid, psiDimBlock, d_density, d_evenPsiPara, dimensions, bodies, volume);
#endif
		constexpr myFloat ABSOLUTE_MARGIN = 1.05;
		
		if (t > 0 && dens > ABSOLUTE_MARGIN || isnan(dens))
		{
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
#if COMPUTE_ERROR || ANALYTIC
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

			if (!dtIncreaseCount)
			{
				dt -= dt_decrese;
				return 0;
			}
			else
			{
				std::cout << dt - dt_increse << ", ";
				return 1;
			}
		}
#endif
#else
#if COMPUTE_COM
#if HYPERBOLIC
		myFloat3 comHyper = centerOfMass(dimGrid, psiDimBlock, d_com, d_oddPsiHyper, dimensions, bodies, volume, block_scale, d_p0);
		std::cout << comHyper.x << ", ";
#endif
#if PARABOLIC
		myFloat3 comPara = centerOfMass(dimGrid, psiDimBlock, d_com, d_oddPsiPara, dimensions, bodies, volume, block_scale, d_p0);
		std::cout << comPara.x << ", "; // << std::endl;
#endif
#endif
#endif

		// integrate one iteration
#if COMPUTE_STABLE_DT
		for (uint step = 0; step < 200; step++) // For checking the numerical stability (not dependent on dt)
#else
		for (uint step = 0; step < IMAGE_SAVE_FREQUENCY; step++)
#endif
		{
			// update odd values (imaginary terms)
#if 1// ANALYTIC
			phaseTime += dt;
#endif
			t += dt / omega_r * 1e3; // [ms]
			signal = getSignal(t);
			Bs.Bq = BqScale * signal.Bq;
			Bs.Bb = BzScale * signal.Bb;
			Bs.BqQuad = BqQuadScale * signal.Bq;
			Bs.BbQuad = BzQuadScale * signal.Bb;

#if HYPERBOLIC
			update_psi << <dimGrid, psiDimBlock >> > (d_oddPsiHyper, d_evenPsiHyper, d_evenQHyper, d_d1, d_hodges, Bs, dimensions, block_scale, d_p0, c0, c2, dt, true, tau);
			update_q_hyper << <dimGrid, edgeDimBlock >> > (d_oddQHyper, d_evenQHyper, d_evenPsiHyper, d_d0, dimensions, dt_per_tau);
#endif
#if PARABOLIC
			update_psi << <dimGrid, psiDimBlock >> > (d_oddPsiPara, d_evenPsiPara, d_evenQPara, d_d1, d_hodges, Bs, dimensions, block_scale, d_p0, c0, c2, dt, false, 0);
			update_q_para << <dimGrid, edgeDimBlock >> > (d_oddQPara, d_oddPsiPara, d_d0, dimensions);
#endif
			// update even values (real terms)
#if 1// ANALYTIC
			phaseTime += dt;
#endif
			t += dt / omega_r * 1e3; // [ms]
			signal = getSignal(t);
			Bs.Bq = BqScale * signal.Bq;
			Bs.Bb = BzScale * signal.Bb;
			Bs.BqQuad = BqQuadScale * signal.Bq;
			Bs.BbQuad = BzQuadScale * signal.Bb;
			
#if HYPERBOLIC
			update_psi << <dimGrid, psiDimBlock >> > (d_evenPsiHyper, d_oddPsiHyper, d_oddQHyper, d_d1, d_hodges, Bs, dimensions, block_scale, d_p0, c0, c2, dt, true, tau);
			update_q_hyper << <dimGrid, edgeDimBlock >> > (d_evenQHyper, d_oddQHyper, d_oddPsiHyper, d_d0, dimensions, dt_per_tau);
#endif
#if PARABOLIC
			update_psi << <dimGrid, psiDimBlock >> > (d_evenPsiPara, d_oddPsiPara, d_oddQPara, d_d1, d_hodges, Bs, dimensions, block_scale, d_p0, c0, c2, dt, false, 0);
			update_q_para << <dimGrid, edgeDimBlock >> > (d_evenQPara, d_evenPsiPara, d_d0, dimensions);
#endif
			iterCount += 2;
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
		myFloat2 hError = { 0 };
		//myFloat hError = { 0 };
		checkCudaErrors(cudaMemcpy(&hError, d_error, sizeof(myFloat2), cudaMemcpyDeviceToHost));
		//std::cout << getDensity(dimGrid, psiDimBlock, d_density, d_evenPsiPara, dimensions, bodies, volume) - sqrt((conj(hError) * hError).x) << ", ";
		//std::cout << getDensity(dimGrid, psiDimBlock, d_density, d_evenPsiPara, dimensions, bodies, volume) - sqrt((conj(hError) * hError).x) << std::endl;
		std::cout << hError.x << ", " << hError.y << "; " << std::endl;
		//std::cout << getDensity(dimGrid, psiDimBlock, d_density, d_evenPsiPara, dimensions, bodies, volume) - hError.x << ", ";
#endif
#if ANALYTIC
		// Compute error
		myFloat2 phaseShift = myFloat2{ cos(-phaseTime * E), sin(-phaseTime * E) };
		analyticStep << <dimGrid, psiDimBlock >> > (d_analyticPsi, d_groundPsi, dimensions, phaseShift);

#if HYPERBOLIC
		weightedDiff << <dimGrid, psiDimBlock >> > (d_error, d_analyticPsi, d_oddPsiHyper, dimensions);
#elif PARABOLIC
		weightedDiff << <dimGrid, psiDimBlock >> > (d_error, d_analyticPsi, d_oddPsiPara, dimensions);
#endif
		int prevStride = bodies;
		while (prevStride > 1)
		{
			int newStride = prevStride / 2;
			integrate << <dim3(std::ceil(newStride / 32.0), 1, 1), dim3(32, 1, 1) >> > (d_error, newStride, ((newStride * 2) != prevStride), volume);
			prevStride = newStride;
		}
		myFloat2 hError = { 0 };
		//myFloat hError = { 0 };
		checkCudaErrors(cudaMemcpy(&hError, d_error, sizeof(myFloat2), cudaMemcpyDeviceToHost));
		//std::cout << getDensity(dimGrid, psiDimBlock, d_density, d_evenPsiPara, dimensions, bodies, volume) - sqrt((conj(hError) * hError).x) << ", ";
		std::cout << hError.x << ", "; // << hError.y << "; ";
		//std::cout << getDensity(dimGrid, psiDimBlock, d_density, d_evenPsiPara, dimensions, bodies, volume) - hError.x << ", ";
#endif
		//std::cout << t << ", ";

#if COMPUTE_GROUND_STATE
		// Copy back from device memory to host memory
		checkCudaErrors(cudaMemcpy3D(&oddPsiBackParamsHyper));

		myFloat2 start = { 1, 0 };
		myFloat startPhase = atan2(start.y, start.x);

		myFloat2 temp = h_oddPsiHyper[xsize * ysize * zsize / 2].values[5].s0;
		myFloat endPhase = atan2(temp.y, temp.x);
		myFloat phaseDiff = endPhase - startPhase;
		std::cout << "Energy was " << phaseDiff / phaseTime << std::endl;
#endif
#if SAVE_STATES
		// Copy back from device memory to host memory
		checkCudaErrors(cudaMemcpy3D(&oddPsiBackParamsHyper));

		savePreImageSpinor(vtks_folder, h_oddPsiHyper, bsize, dxsize, dysize, dzsize, block_scale, d_p0, t);

		if (((int(t) % STATE_SAVE_INTERVAL) == 0) && (int(t) != lastSaveTime))
		{
			std::ofstream oddFs(toString(t) + ".dat", std::ios::binary | std::ios_base::trunc);
			if (oddFs.fail() != 0) return 1;
			oddFs.write((char*)&h_oddPsiHyper[0], hostSize * sizeof(BlockPsis));
			oddFs.close();

			checkCudaErrors(cudaMemcpy3D(&evenPsiBackParamsHyper));
			std::ofstream evenFs("even_" + toString(t) + ".dat", std::ios::binary | std::ios_base::trunc);
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
	cudaDeviceSynchronize();
	auto duration = std::chrono::high_resolution_clock::now() - prevTime;
	//std::cout << "Simulation time: " << t << " ms. Real time: " << duration.count() * 1e-9 << " s." << std::endl;

	checkCudaErrors(cudaMemcpy3D(&oddPsiBackParamsHyper));
	drawDensity("hyper_verify", h_oddPsiHyper, dxsize, dysize, dzsize, t, ".");
#elif PARABOLIC
	cudaDeviceSynchronize();
	auto duration = std::chrono::high_resolution_clock::now() - prevTime;
	//std::cout << "Simulation time: " << t << " ms. Real time: " << duration.count() * 1e-9 << " s." << std::endl;

	checkCudaErrors(cudaMemcpy3D(&oddPsiBackParamsPara));
	drawDensity("para_verify", h_oddPsiPara, dxsize, dysize, dzsize, t, ".");
#endif
#if COMPUTE_ERROR
	std::cout << "]';" << std::endl;
#endif
	//std::cout << iterCount << " iterations" << std::endl;
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
#if COMPUTE_ERROR || ANALYTIC
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
#if COMPUTE_STABLE_DT
	dt += dt_increse;
	dtIncreaseCount++;
#endif
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
				dt_per_tau = dt / tau;
			}
			else if (size_t pos = line.find("tau") != std::string::npos)
			{
				tau = std::stod(line.substr(pos + 5));
				dt_per_tau = dt / tau;
			}
		}
	}
}

int main(int argc, char** argv)
{
#if PARABOLIC && HYPERBOLIC
	std::cout << "Computing both parabolic and hyperbolic" << std::endl << std::endl;
#elif PARABOLIC
	std::cout << "Computing parabolic" << std::endl << std::endl;
#elif HYPERBOLIC
	std::cout << "Computing hyperbolic" << std::endl << std::endl;
#endif

	readConfFile();

	

	std::cout << "Start simulating from t = " << t << " ms." << std::endl;
	std::cout << "The simulation will end at " << END_TIME << " ms." << std::endl;
	//std::cout << "Block scale = " << blockScale << std::endl;
	std::cout << "dt = " << dt << " = " << dt / omega_r * 1e3 << " ms." << std::endl;
	std::cout << "tau = " << tau << std::endl;
	std::cout << "dt / tau = " << dt_per_tau << std::endl;

	// integrate in time using DEC
	auto domainMin = Vector3(-DOMAIN_SIZE_X * 0.5, -DOMAIN_SIZE_Y * 0.5, -DOMAIN_SIZE_Z * 0.5);
	auto domainMax = Vector3(DOMAIN_SIZE_X * 0.5, DOMAIN_SIZE_Y * 0.5, DOMAIN_SIZE_Z * 0.5);

#if COMPUTE_STABLE_DT
	for (int i = 0; i < 10; ++i)  
	{
		REPLICABLE_STRUCTURE_COUNT_X = 58.0 + i * 6.0;
		const myFloat blockScale = DOMAIN_SIZE_X / REPLICABLE_STRUCTURE_COUNT_X / BLOCK_WIDTH_X;
	
		while (!integrateInTime(blockScale, domainMin, domainMax))
		{
			dt_per_tau = dt / tau;
			IMAGE_SAVE_FREQUENCY = uint(IMAGE_SAVE_INTERVAL * 0.5 / 1e3 * omega_r / dt) + 1;
			t = 0;
		}
		dt_per_tau = dt / tau;
		IMAGE_SAVE_FREQUENCY = uint(IMAGE_SAVE_INTERVAL * 0.5 / 1e3 * omega_r / dt) + 1;
		t = 0;
		
		dtIncreaseCount = 0;
	}
#else
	const myFloat blockScale = DOMAIN_SIZE_X / REPLICABLE_STRUCTURE_COUNT_X / BLOCK_WIDTH_X;
	std::cout << "Dual edge length = " << DUAL_EDGE_LENGTH * blockScale << std::endl;
#if COMPUTE_COM
	std::cout << "corrected_F1_com = [";
	//while (tau < 0.0031)
	{
		t = 0;
		integrateInTime(blockScale, domainMin, domainMax);
		std::cout << ";";
		//tau += 0.00025;
		//dt_per_tau = dt / tau;
	}
	std::cout << "];" << std::endl;
#else
	integrateInTime(blockScale, domainMin, domainMax);
#endif
#endif
	return 0;
}
