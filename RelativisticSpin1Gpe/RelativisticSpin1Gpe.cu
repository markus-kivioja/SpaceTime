#include <cuda_runtime.h>
#include "helper_cuda.h"

constexpr double CREATION_RAMP_START = 0.1;
constexpr double EXPANSION_START = CREATION_RAMP_START + 10.5; // When the expansion starts in ms

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

#define RELATIVISTIC 1

#define COMPUTE_GROUND_STATE 0

#define SAVE_STATES 0
#define SAVE_PICTURE 1

#define THREAD_BLOCK_X 16
#define THREAD_BLOCK_Y 2
#define THREAD_BLOCK_Z 1

constexpr double DOMAIN_SIZE_X = 24.0;
constexpr double DOMAIN_SIZE_Y = 24.0;
constexpr double DOMAIN_SIZE_Z = 24.0;

constexpr double REPLICABLE_STRUCTURE_COUNT_X = 112.0;
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

constexpr double myGamma = 2.9e-30;
const double alpha = N * N * myGamma * 1e-12 / (a_r * a_r * a_r * a_r * a_r * a_r * 2 * PI * trapFreq_r);

constexpr double muB = 9.27400968e-24; // [m^2 kg / s^2 T^-1] Bohr magneton

const double BqScale = -(0.5 * muB / (hbar * omega_r) * a_r) / 100.; // [cm/Gauss]
constexpr double BzScale = -(0.5 * muB / (hbar * omega_r)) / 10000.; // [1/Gauss]

constexpr double A_hfs = 3.41734130545215;
const double BqQuadScale = 100 * a_r * sqrt(0.25 * 1000 * (1.399624624 * 1.399624624) / (trapFreq_r * 2 * A_hfs)); //[cm/Gauss]
const double BzQuadScale = sqrt(0.25 * 1000 * (1.399624624 * 1.399624624) / (trapFreq_r * 2 * A_hfs)); //[1/Gauss]  \sqrt{g_q}

constexpr double SQRT_2 = 1.41421356237309;
constexpr double INV_SQRT_2 = 0.70710678118655;

constexpr double NOISE_AMPLITUDE = 0.1;

double para_dt = 7e-4; // Max parabolic: 7e-4
double hyper_dt = para_dt; //3e-3; // Max hyperbolic: 3e-3

const float IMAGE_SAVE_INTERVAL = 0.5; // ms
uint IMAGE_SAVE_FREQUENCY = uint(IMAGE_SAVE_INTERVAL * 0.5 / 1e3 * omega_r / hyper_dt) + 1;

const uint STATE_SAVE_INTERVAL = 10.0; // ms

double t = 0; // Start time in ms
double END_TIME = 0.6; // End time in ms

double sigma = 1.0; // 0.01;
double dt_per_sigma = hyper_dt / sigma;

enum class Phase
{
	Polar = 0,
	Ferromagnetic
};
constexpr Phase initPhase = Phase::Polar;
//constexpr Phase initPhase = Phase::Ferromagnetic;

std::string toStringShort(const double value)
{
	std::ostringstream out;
	out.precision(2);
	out << std::fixed << value;
	return out.str();
};

const std::string GROUND_STATE_PSI_FILENAME = "ground_state_psi_" + toStringShort(sigma) + ".dat";
const std::string GROUND_STATE_Q_FILENAME = "ground_state_q_" + toStringShort(sigma) + ".dat";
const std::string GROUND_STATE_HYPER_FILENAME = "ground_state.dat";

#include "para_kernels.h"
#include "hyper_kernels.h"
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

void printDensity(dim3 dimGrid, dim3 dimBlock, double* densityPtr, PitchedPtr psi, uint3 dimensions, size_t bodies, double volume)
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

	std::cout << "Total density: " << hDensity << std::endl;
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

template<typename T>
T* allocHost(size_t count)
{
	T* ptr;
	checkCudaErrors(cudaMallocHost(&ptr, count * sizeof(T)));
	memset(ptr, 0, count * sizeof(T));
	return ptr;
}

cudaPitchedPtr allocDevice3D(cudaExtent extent)
{
	cudaPitchedPtr ptr;
	checkCudaErrors(cudaMalloc3D(&ptr, extent));
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
	//std::cout << "Replicable structure instances in x: " << xsize << ", y: " << ysize << ", z: " << zsize << std::endl;
	uint64_t bodies = xsize * ysize * zsize * bsize;
	//std::cout << "Dual 0-cells in total: " << bodies << std::endl;

	// Initialize device memory
	size_t dxsize = xsize + 2; // One element buffer to both ends
	size_t dysize = ysize + 2; // One element buffer to both ends
	size_t dzsize = zsize + 2; // One element buffer to both ends
	cudaExtent psiExtent = make_cudaExtent(dxsize * sizeof(BlockPsis), dysize, dzsize);
	cudaExtent edgeExtent = make_cudaExtent(dxsize * sizeof(BlockEdges), dysize, dzsize);

	cudaPitchedPtr d_cudaEvenPsi = allocDevice3D(psiExtent);
	cudaPitchedPtr d_cudaEvenQ = allocDevice3D(edgeExtent);
	cudaPitchedPtr d_cudaOddPsi = allocDevice3D(psiExtent);
	cudaPitchedPtr d_cudaOddQ = allocDevice3D(edgeExtent);

	// For computing the energy/chemical potential
	cudaPitchedPtr d_cudaHPsi = allocDevice3D(edgeExtent);

	double* d_spinNorm = allocDevice<double>(bodies);
	double* d_density = allocDevice<double>(bodies);
	double* d_energy = allocDevice<double>(bodies);
	double3* d_localAvgSpin = allocDevice<double3>(bodies);
	double3* d_u = allocDevice<double3>(bodies);
	double3* d_v = allocDevice<double3>(bodies);
	double* d_theta = allocDevice<double>(bodies);

	size_t offset = d_cudaEvenPsi.pitch * dysize + d_cudaEvenPsi.pitch + sizeof(BlockPsis);
	size_t edgeOffset = d_cudaEvenQ.pitch * dysize + d_cudaEvenQ.pitch + sizeof(BlockEdges);
	PitchedPtr d_evenPsi = { (char*)d_cudaEvenPsi.ptr + offset, d_cudaEvenPsi.pitch, d_cudaEvenPsi.pitch * dysize };
	PitchedPtr d_evenQ = { (char*)d_cudaEvenQ.ptr + edgeOffset, d_cudaEvenQ.pitch, d_cudaEvenQ.pitch * dysize };
	PitchedPtr d_oddPsi = { (char*)d_cudaOddPsi.ptr + offset, d_cudaOddPsi.pitch, d_cudaOddPsi.pitch * dysize };
	PitchedPtr d_oddQ = { (char*)d_cudaOddQ.ptr + edgeOffset, d_cudaOddQ.pitch, d_cudaOddQ.pitch * dysize };

	PitchedPtr d_HPsi = { (char*)d_cudaHPsi.ptr + offset, d_cudaHPsi.pitch, d_cudaHPsi.pitch * dysize };

	// find terms for laplacian
	Buffer<int3> d0;
	Buffer<int2> d1;
	Buffer<double> hodges;
	getLaplacian(hodges, d0, d1, sizeof(BlockPsis), d_evenPsi.pitch, d_evenPsi.slicePitch, sizeof(BlockEdges), d_evenQ.pitch, d_evenQ.slicePitch);

	//std::cout << "lapsize = " << lapsize << ", lapfac = " << lapfac << ", lapfac0 = " << lapfac0 << std::endl;

	for (int i = 0; i < hodges.size(); ++i) hodges[i] = -0.5 * hodges[i] / (block_scale * block_scale);

	int3* d_d0;
	checkCudaErrors(cudaMalloc(&d_d0, d0.size() * sizeof(int3)));

	int2* d_d1;
	checkCudaErrors(cudaMalloc(&d_d1, d1.size() * sizeof(int2)));

	double* d_hodges;
	checkCudaErrors(cudaMalloc(&d_hodges, hodges.size() * sizeof(double)));

	// Initialize host memory
	size_t hostSize = dxsize * dysize * dzsize;
	BlockPsis* h_evenPsi = allocHost<BlockPsis>(hostSize);
	BlockPsis* h_oddPsi = allocHost<BlockPsis>(hostSize);
	BlockEdges* h_evenQ = allocHost<BlockEdges>(hostSize);
	BlockEdges* h_oddQ = allocHost<BlockEdges>(hostSize);

	double* h_density = allocHost<double>(bodies);
	double3* h_u = allocHost<double3>(bodies);
	double* h_theta = allocHost<double>(bodies);
	double3* h_localAvgSpin = allocHost<double3>(bodies);

#if COMPUTE_GROUND_STATE
	// Initialize discrete field
	std::ifstream fs(GROUND_STATE_PSI_FILENAME, std::ios::binary | std::ios::in);
	bool continueFromEarlier = (fs.fail() == 0);
	if (continueFromEarlier)
	{
		std::cout << "Initialized ground state psi from file." << std::endl;

		fs.read((char*)&h_evenPsi[0], hostSize * sizeof(BlockPsis));
		fs.close();

		std::ifstream fs_q(GROUND_STATE_Q_FILENAME, std::ios::binary | std::ios::in);
		if (fs.fail() == 0)
		{
			std::cout << "Initialized ground state q from file." << std::endl;

			fs.read((char*)&h_evenQ[0], hostSize * sizeof(BlockEdges));
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
						h_evenPsi[dstI].values[l].s1 = s1;
						h_evenPsi[dstI].values[l].s0 = s0;
						h_evenPsi[dstI].values[l].s_1 = s_1;
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
	std::ifstream psi_fs(psi_filename, std::ios::binary | std::ios::in);
	if (psi_fs.fail() != 0)
	{
		std::cout << "Failed to open file " << psi_filename << std::endl;
		return 1;
	}
	else
	{
		std::cout << "Loading ground state psi from " << psi_filename << "..." << std::endl;
	}
	psi_fs.read((char*)&h_oddPsi[0], hostSize * sizeof(BlockPsis));
	psi_fs.close();

#if RELATIVISTIC
	std::string q_filename = loadGroundState ? GROUND_STATE_Q_FILENAME : toString(t) + ".dat";
	std::ifstream q_fs(q_filename, std::ios::binary | std::ios::in);
	if (q_fs.fail() != 0)
	{
		std::cout << "Failed to open file " << q_filename << std::endl;
		return 1;
	}
	else
	{
		std::cout << "Loading ground state q from " << q_filename << "..." << std::endl;
	}
	q_fs.read((char*)&h_oddQ[0], hostSize * sizeof(BlockEdges));
	q_fs.close();
#endif

	bool doForward = true;
	std::string evenFilename = "even_" + toString(t) + ".dat";
	std::ifstream evenFs(evenFilename, std::ios::binary | std::ios::in);
	if (evenFs.fail() == 0)
	{
		evenFs.read((char*)&h_evenPsi[0], hostSize * sizeof(BlockPsis));
		evenFs.close();
		doForward = false;
		std::cout << "Loaded even time step from file" << std::endl;
	}

#endif

	cudaPitchedPtr h_cudaEvenPsi = copyHostToDevice3D(h_evenPsi, d_cudaEvenPsi, psiExtent);
	cudaPitchedPtr h_cudaOddPsi = copyHostToDevice3D(h_oddPsi, d_cudaOddPsi, psiExtent);
	cudaPitchedPtr h_cudaEvenQ = copyHostToDevice3D(h_evenQ, d_cudaEvenQ, edgeExtent);
	cudaPitchedPtr h_cudaOddQ = copyHostToDevice3D(h_oddQ, d_cudaOddQ, edgeExtent);

	checkCudaErrors(cudaMemcpy(d_d0, &d0[0], d0.size() * sizeof(int3), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_d1, &d1[0], d1.size() * sizeof(int2), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_hodges, &hodges[0], hodges.size() * sizeof(double), cudaMemcpyHostToDevice));

	// Clear host memory after data has been copied to devices
	cudaDeviceSynchronize();
	d0.clear();
	d1.clear();
	hodges.clear();
#if !(SAVE_PICTURE)
	cudaFreeHost(h_evenPsi);
	cudaFreeHost(h_oddPsi);
#endif
	cudaMemcpy3DParms evenPsiBackParams = createDeviceToHostParams(d_cudaEvenPsi, h_cudaEvenPsi, psiExtent);
	cudaMemcpy3DParms oddPsiBackParams = createDeviceToHostParams(d_cudaOddPsi, h_cudaOddPsi, psiExtent);
	cudaMemcpy3DParms evenQBackParams = createDeviceToHostParams(d_cudaEvenQ, h_cudaEvenQ, edgeExtent);
	cudaMemcpy3DParms oddQBackParams = createDeviceToHostParams(d_cudaOddQ, h_cudaOddQ, edgeExtent);

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

	if (loadGroundState)
	{
		switch (initPhase)
		{
		case Phase::Polar:
			std::cout << "Transform ground state to polar phase" << std::endl;
			polarState << <dimGrid, psiDimBlock >> > (d_oddPsi, dimensions);
			break;
		case Phase::Ferromagnetic:
			std::cout << "Transform ground state to ferromagnetic phase" << std::endl;
			ferromagneticState << <dimGrid, psiDimBlock >> > (d_oddPsi, dimensions);
			break;
		default:
			break;
		}

		printDensity(dimGrid, psiDimBlock, d_density, d_oddPsi, dimensions, bodies, volume);
	}

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
		forwardEuler << <dimGrid, psiDimBlock >> > (d_evenPsi, d_oddPsi, d_oddQ, d_d1, d_hodges, Bs, dimensions, block_scale, d_p0, c0, c2, hyper_dt);
		forwardEuler_q_hyper << <dimGrid, edgeDimBlock >> > (d_evenQ, d_oddQ, d_oddPsi, d_d0, dimensions, dt_per_sigma);

		// Parabolic
		//forwardEuler_q_para << <dimGrid, edgeDimBlock >> > (d_oddQ, d_oddQ, d_oddPsi, d_d0, dimensions);
		//forwardEuler_para << <dimGrid, psiDimBlock >> > (d_evenPsi, d_oddPsi, d_oddQ, d_d1, d_hodges, Bs, dimensions, block_scale, d_p0, c0, c2, para_dt);
		//forwardEuler_q_para << <dimGrid, edgeDimBlock >> > (d_evenQ, d_evenQ, d_evenPsi, d_d0, dimensions);
	}
	else
#endif
	{
		std::cout << "Skipping the forward step." << std::endl;
	}

#if COMPUTE_GROUND_STATE
	std::string folder = "gs_dens_profiles";
	std::string createResultsDirCommand = "mkdir " + folder;
	system(createResultsDirCommand.c_str());

	uint iter = 0;
	
	if (!continueFromEarlier)
	{
		normalize_h(dimGrid, psiDimBlock, d_density, d_evenPsi, dimensions, bodies, volume);
		normalize_h(dimGrid, psiDimBlock, d_density, d_oddPsi, dimensions, bodies, volume);
		itp_q << <dimGrid, edgeDimBlock >> > (d_evenQ, d_evenQ, d_evenPsi, d_d0, dimensions, dt_per_sigma);
		itp_q << <dimGrid, edgeDimBlock >> > (d_oddQ, d_oddQ, d_oddPsi, d_d0, dimensions, dt_per_sigma);
	}

	while (true)
	{
		if ((iter % 1000) == 0) std::cout << "Iteration " << iter << std::endl;
#if SAVE_PICTURE
		if ((iter % 1000) == 0)
		{
			checkCudaErrors(cudaMemcpy3D(&evenPsiBackParams));
			drawDensity(h_evenPsi, dxsize, dysize, dzsize, iter, folder);
			printDensity(dimGrid, psiDimBlock, d_density, d_evenPsi, dimensions, bodies, volume);

			// Compute energy/chemical potential
			innerProduct << <dimGrid, psiDimBlock >> > (d_energy, d_evenPsi, d_HPsi, dimensions);
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
			checkCudaErrors(cudaMemcpy3D(&evenPsiBackParams));
			std::ofstream fs_psi(GROUND_STATE_PSI_FILENAME, std::ios::binary | std::ios_base::trunc);
			if (fs_psi.fail() != 0) return 1;
			fs_psi.write((char*)&h_evenPsi[0], hostSize * sizeof(BlockPsis));
			fs_psi.close();

			// Q
			checkCudaErrors(cudaMemcpy3D(&evenQBackParams));
			std::ofstream fs_q(GROUND_STATE_Q_FILENAME, std::ios::binary | std::ios_base::trunc);
			if (fs_q.fail() != 0) return 1;
			fs_q.write((char*)&h_evenQ[0], hostSize * sizeof(BlockEdges));
			fs_q.close();

			return 0;
		}
#if RELATIVISTIC
		// Take an imaginary time step
		itp_q << <dimGrid, edgeDimBlock >> > (d_oddQ, d_evenQ, d_evenPsi, d_d0, dimensions, dt_per_sigma);
		itp_psi << <dimGrid, psiDimBlock >> > (d_HPsi, d_oddPsi, d_evenPsi, d_evenQ, d_d1, d_hodges, Bs, dimensions, block_scale, d_p0, c0, c2, dt);
		// Normalize
		normalize_h(dimGrid, psiDimBlock, d_density, d_oddPsi, dimensions, bodies, volume);

		// Take an imaginary time step
		itp_q << <dimGrid, edgeDimBlock >> > (d_evenQ, d_oddQ, d_oddPsi, d_d0, dimensions, dt_per_sigma);
		itp_psi << <dimGrid, psiDimBlock >> > (d_HPsi, d_evenPsi, d_oddPsi, d_oddQ, d_d1, d_hodges, Bs, dimensions, block_scale, d_p0, c0, c2, dt);
		// Normalize
		normalize_h(dimGrid, psiDimBlock, d_density, d_evenPsi, dimensions, bodies, volume);
#endif

		iter++;
	}

#else
	std::string times = std::string("times = [times");
	std::string bqString = std::string("Bq = [Bq");
	std::string bzString = std::string("Bz = [Bz");
	std::string spinString = std::string("Spin = [Spin");
	std::string magX = std::string("mag_x = [mag_x");
	std::string magY = std::string("mag_y = [mag_y");
	std::string magZ = std::string("mag_z = [mag_z");
	std::string densityStr = std::string("norm = [norm");

	int lastSaveTime = 0;

	std::string hyper_folder = "hyperbolic";
	std::string para_folder = "parabolic";

	std::string createResultsDirCommand = "mkdir " + hyper_folder;
	system(createResultsDirCommand.c_str());

	while (t < END_TIME)
	{
#if SAVE_PICTURE
		// Copy back from device memory to host memory
		checkCudaErrors(cudaMemcpy3D(&oddPsiBackParams));

		// Measure wall clock time
		static auto prevTime = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::high_resolution_clock::now() - prevTime;
		std::cout << "Simulation time: " << t << " ms. Real time from previous save: " << duration.count() * 1e-9 << " s." << std::endl;
		prevTime = std::chrono::high_resolution_clock::now();

		drawDensity(h_oddPsi, dxsize, dysize, dzsize, t, hyper_folder);

		//uvTheta << <dimGrid, dimBlock >> > (d_u, d_v, d_theta, d_oddPsi, dimensions);
		//cudaMemcpy(h_u, d_u, bodies * sizeof(double3), cudaMemcpyDeviceToHost);
		//cudaMemcpy(h_theta, d_theta, bodies * sizeof(double), cudaMemcpyDeviceToHost);
		//drawUtheta(h_u, h_theta, xsize, ysize, zsize, t - 202.03);
		//
		//ferromagneticDomain << <dimGrid, dimBlock >> > (d_ferroDom, d_oddPsi, dimensions);
		//cudaMemcpy(h_ferroDom, d_ferroDom, bodies * sizeof(double), cudaMemcpyDeviceToHost);
		//drawFerroDom(h_ferroDom, xsize, ysize, zsize, t - 202.03);
#endif
		const uint centerIdx = 57 * dxsize * dysize + 57 * dxsize + 57;
		double2 temp = h_oddPsi[centerIdx].values[5].s0;
		double startPhase = atan2(temp.y, temp.x);
		double phaseTime = 0;

		// integrate one iteration
		for (uint step = 0; step < IMAGE_SAVE_FREQUENCY; step++)
		{
			// update odd values (imaginary terms)
			phaseTime += hyper_dt;
			t += hyper_dt / omega_r * 1e3; // [ms]
			signal = getSignal(t);
			Bs.Bq = BqScale * signal.Bq;
			Bs.Bb = BzScale * signal.Bb;
			Bs.BqQuad = BqQuadScale * signal.Bq;
			Bs.BbQuad = BzQuadScale * signal.Bb;
			update_psi << <dimGrid, psiDimBlock >> > (d_oddPsi, d_evenPsi, d_evenQ, d_d1, d_hodges, Bs, dimensions, block_scale, d_p0, c0, c2, hyper_dt);
			update_q_hyper << <dimGrid, edgeDimBlock >> > (d_oddQ, d_evenQ, d_evenPsi, d_d0, dimensions, dt_per_sigma);

			// Parabolic
			// update_psi << <dimGrid, psiDimBlock >> > (d_oddPsi, d_evenPsi, d_evenQ, d_d1, d_hodges, Bs, dimensions, block_scale, d_p0, c0, c2, parabolic_dt);
			//update_q_para << <dimGrid, edgeDimBlock >> > (d_oddQ, d_oddQ, d_oddPsi, d_d0, dimensions);

			// update even values (real terms)
			phaseTime += hyper_dt;
			t += hyper_dt / omega_r * 1e3; // [ms]
			signal = getSignal(t);
			Bs.Bq = BqScale * signal.Bq;
			Bs.Bb = BzScale * signal.Bb;
			Bs.BqQuad = BqQuadScale * signal.Bq;
			Bs.BbQuad = BzQuadScale * signal.Bb;
			update_psi << <dimGrid, psiDimBlock >> > (d_evenPsi, d_oddPsi, d_oddQ, d_d1, d_hodges, Bs, dimensions, block_scale, d_p0, c0, c2, hyper_dt);
			update_q_hyper << <dimGrid, edgeDimBlock >> > (d_evenQ, d_oddQ, d_oddPsi, d_d0, dimensions, dt_per_sigma);
		
			// Parabolic
			//update_psi << <dimGrid, psiDimBlock >> > (d_evenPsi, d_oddPsi, d_oddQ, d_d1, d_hodges, Bs, dimensions, block_scale, d_p0, c0, c2, para_dt);
			//update_q_para << <dimGrid, edgeDimBlock >> > (d_evenQ, d_evenQ, d_evenPsi, d_d0, dimensions);
		}
		// Copy back from device memory to host memory
		checkCudaErrors(cudaMemcpy3D(&oddPsiBackParams));

		temp = h_oddPsi[centerIdx].values[5].s0;
		double endPhase = atan2(temp.y, temp.x);
		double phaseDiff = endPhase - startPhase;
		std::cout << "Energy was " << phaseDiff / phaseTime << std::endl;

#if SAVE_STATES
		// Copy back from device memory to host memory
		checkCudaErrors(cudaMemcpy3D(&oddPsiBackParams));

		localAvgSpinAndDensity << <dimGrid, psiDimBlock >> > (d_spinNorm, d_localAvgSpin, d_density, d_oddPsi, dimensions);
		cudaMemcpy(h_localAvgSpin, d_localAvgSpin, bodies * sizeof(double3), cudaMemcpyDeviceToHost);
		uvTheta << <dimGrid, psiDimBlock >> > (d_u, d_v, d_theta, d_oddPsi, dimensions);
		cudaMemcpy(h_u, d_u, bodies * sizeof(double3), cudaMemcpyDeviceToHost);
		cudaMemcpy(h_theta, d_theta, bodies * sizeof(double), cudaMemcpyDeviceToHost);
		saveVolume(SAVE_FILE_PREFIX, h_oddPsi, h_localAvgSpin, h_u, h_theta, bsize, dxsize, dysize, dzsize, 0, block_scale, d_p0, t - 202.03);

		SpinMagDens spinMagDens = integrateSpinAndDensity(dimGrid, psiDimBlock, d_spinNorm, d_localAvgSpin, d_density, bodies, volume);
		times += ", " + toString(t);
		bqString += ", " + toString(Bs.Bq);
		bzString += ", " + toString(Bs.Bz);
		spinString += ", " + toString(spinMagDens.spin);
		magX += ", " + toString(spinMagDens.magnetization.x);
		magY += ", " + toString(spinMagDens.magnetization.y);
		magZ += ", " + toString(spinMagDens.magnetization.z);
		densityStr += ", " + toString(spinMagDens.density);

		if (((int(t) % STATE_SAVE_INTERVAL) == 0) && (int(t) != lastSaveTime))
		{
			times += "];";
			bqString += "];";
			bzString += "];";
			spinString += "];";
			magX += "];";
			magY += "];";
			magZ += "];";
			densityStr += "];";

			Text textFile;
			textFile << times << std::endl;
			textFile << bqString << std::endl;
			textFile << bzString << std::endl;
			textFile << spinString << std::endl;
			textFile << magX << std::endl;
			textFile << magY << std::endl;
			textFile << magZ << std::endl;
			textFile << densityStr << std::endl;
			textFile.save(SAVE_FILE_PREFIX + toString(t) + ".m");

			std::ofstream oddFs(SAVE_FILE_PREFIX + toString(t) + ".dat", std::ios::binary | std::ios_base::trunc);
			if (oddFs.fail() != 0) return 1;
			oddFs.write((char*)&h_oddPsi[0], hostSize * sizeof(BlockPsis));
			oddFs.close();

			checkCudaErrors(cudaMemcpy3D(&evenPsiBackParams));
			std::ofstream evenFs(SAVE_FILE_PREFIX + "even_" + toString(t) + ".dat", std::ios::binary | std::ios_base::trunc);
			if (evenFs.fail() != 0) return 1;
			evenFs.write((char*)&h_evenPsi[0], hostSize * sizeof(BlockPsis));
			evenFs.close();

			std::cout << "Saved the state!" << std::endl;

			times = std::string("times = [times");
			bqString = std::string("Bq = [Bq");
			bzString = std::string("Bz = [Bz");
			spinString = std::string("Spin = [Spin");
			magX = std::string("mag_x = [mag_x");
			magY = std::string("mag_y = [mag_y");
			magZ = std::string("mag_z = [mag_z");
			densityStr = std::string("norm = [norm");

			lastSaveTime = int(t);
		}
#endif
	}
#endif

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
				hyper_dt = std::stod(line.substr(pos + 2));
				IMAGE_SAVE_FREQUENCY = uint(IMAGE_SAVE_INTERVAL * 0.5 / 1e3 * omega_r / hyper_dt) + 1;
				dt_per_sigma = hyper_dt / sigma;
			}
			else if (size_t pos = line.find("sigma") != std::string::npos)
			{
				sigma = std::stod(line.substr(pos + 5));
				dt_per_sigma = hyper_dt / sigma;
			}
		}
	}
}

int main(int argc, char** argv)
{
	readConfFile();

	const double blockScale = DOMAIN_SIZE_X / REPLICABLE_STRUCTURE_COUNT_X / BLOCK_WIDTH_X;

	std::cout << "Start simulating from t = " << t << " ms, with a time step size of " << hyper_dt << "." << std::endl;
	std::cout << "The simulation will end at " << END_TIME << " ms." << std::endl;
	//std::cout << "Block scale = " << blockScale << std::endl;
	//std::cout << "Dual edge length = " << DUAL_EDGE_LENGTH * blockScale << std::endl;
	std::cout << "Relativistic sigma = " << sigma << std::endl;

	// integrate in time using DEC
	auto domainMin = Vector3(-DOMAIN_SIZE_X * 0.5, -DOMAIN_SIZE_Y * 0.5, -DOMAIN_SIZE_Z * 0.5);
	auto domainMax = Vector3(DOMAIN_SIZE_X * 0.5, DOMAIN_SIZE_Y * 0.5, DOMAIN_SIZE_Z * 0.5);
	integrateInTime(blockScale, domainMin, domainMax);

	return 0;
}
