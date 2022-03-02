#include <cuda_runtime.h>
#include <helper_cuda.h>

#include "AliceRingRamps.h"

#include "VortexState.hpp"
#include <Output/Picture.hpp>
#include <Output/Text.hpp>
#include <Types/Complex.hpp>
#include <Types/Random.hpp>
#include <Mesh/DelaunayMesh.hpp>
#include <iostream>
#include <sstream>
#include <chrono>

#include <mesh.h>

constexpr double Lx = 24.0;
constexpr double Ly = 24.0;
constexpr double Lz = 24.0;

constexpr double Nx = 255.0;
constexpr double Ny = 255.0;
constexpr double Nz = 255.0;

constexpr double N = 2e5;

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

constexpr double gamma = 2.9e-30;
const double alpha = N * N * gamma * 1e-12 / (a_r * a_r * a_r * a_r * a_r * a_r * 2 * PI * trapFreq_r);

constexpr double muB = 9.27400968e-24; // [m^2 kg / s^2 T^-1] Bohr magneton

const double BqScale = -(0.5 * muB / (hbar * omega_r) * a_r) / 100.; // [cm/Gauss]
constexpr double BzScale = -(0.5 * muB / (hbar * omega_r)) / 10000.; // [1/Gauss]

constexpr double A_hfs = 10.11734130545215;
const double BqQuadScale = 100 * a_r * sqrt(0.25 * 1000 * (1.399624624 * 1.399624624) / (trapFreq_r * 2 * A_hfs)); //[cm/Gauss]
const double BzQuadScale = sqrt(0.25 * 1000 * (1.399624624 * 1.399624624) / (trapFreq_r * 2 * A_hfs)); //[1/Gauss]  \sqrt{g_q}

constexpr double dt = 2e-4; // iteration_period / double(steps_per_iteration); // time step in time units

const std::string STATE_FILENAME = "polar_ground_state.dat";

static constexpr double SQRT_2 = 1.41421356237309;
static constexpr double INV_SQRT_2 = 0.70710678118655;

#define COMPUTE_GROUND_STATE 0

#define SAVE_PICTURE 1
#define SAVE_VOLUME 0
#define SAVE_FREQUENCY 1000

#define THREAD_BLOCK_X 8
#define THREAD_BLOCK_Y 8
#define THREAD_BLOCK_Z 1

__host__ __device__ __inline__ double trap(double3 p)
{
	double x = p.x * lambda_x;
	double y = p.y * lambda_y;
	double z = p.z * lambda_z;
	return 0.5 * (x * x + y * y + z * z) + 100.0;
}

__host__ __device__ __inline__ double3 magneticField(double3 p, double Bq, double Bz)
{
	return make_double3(Bq * p.x, Bq * p.y, -2 * Bq * p.z + Bz);
}

#include <utils.h>

__global__ void density(double* density, PitchedPtr prevStep, uint3 dimensions, double dv)
{
	size_t xid = blockIdx.x * blockDim.x + threadIdx.x;
	size_t yid = blockIdx.y * blockDim.y + threadIdx.y;
	size_t zid = blockIdx.z * blockDim.z + threadIdx.z;
	size_t dataZid = zid / VALUES_IN_BLOCK; // One thread per every dual node so VALUES_IN_BLOCK threads per mesh block (on z-axis)

	// Exit leftover threads
	if (xid >= dimensions.x || yid >= dimensions.y || dataZid >= dimensions.z)
	{
		return;
	}

	size_t dualNodeId = zid % VALUES_IN_BLOCK; // Dual node id. One thread per every dual node so VALUES_IN_BLOCK threads per mesh block (on z-axis)

	char* pPsi = prevStep.ptr + prevStep.slicePitch * dataZid + prevStep.pitch * yid + sizeof(BlockPsis) * xid;
	Complex3Vec psi = ((BlockPsis*)pPsi)->values[dualNodeId];

	size_t idx = dataZid * dimensions.x * dimensions.y * VALUES_IN_BLOCK + yid * dimensions.x * VALUES_IN_BLOCK + xid * VALUES_IN_BLOCK + dualNodeId;
	density[idx] = dv * ((psi.s1 * conj(psi.s1)).x + (psi.s0 * conj(psi.s0)).x + (psi.s_1 * conj(psi.s_1)).x);
}

__global__ void magnetizationAndDensity(double3* pMagnetization, double* density, PitchedPtr prevStep, uint3 dimensions, double dv)
{
	size_t xid = blockIdx.x * blockDim.x + threadIdx.x;
	size_t yid = blockIdx.y * blockDim.y + threadIdx.y;
	size_t zid = blockIdx.z * blockDim.z + threadIdx.z;
	size_t dataZid = zid / VALUES_IN_BLOCK; // One thread per every dual node so VALUES_IN_BLOCK threads per mesh block (on z-axis)

	// Exit leftover threads
	if (xid >= dimensions.x || yid >= dimensions.y || dataZid >= dimensions.z)
	{
		return;
	}

	size_t dualNodeId = zid % VALUES_IN_BLOCK; // Dual node id. One thread per every dual node so VALUES_IN_BLOCK threads per mesh block (on z-axis)

	char* pPsi = prevStep.ptr + prevStep.slicePitch * dataZid + prevStep.pitch * yid + sizeof(BlockPsis) * xid;
	Complex3Vec psi = ((BlockPsis*)pPsi)->values[dualNodeId];

	double normSq_s1 = psi.s1.x * psi.s1.x + psi.s1.y * psi.s1.y;
	double normSq_s0 = psi.s0.x * psi.s0.x + psi.s0.y * psi.s0.y;
	double normSq_s_1 = psi.s_1.x * psi.s_1.x + psi.s_1.y * psi.s_1.y;
	double2 temp = SQRT_2 * (conj(psi.s1) * psi.s0 + conj(psi.s0) * psi.s_1);
	double3 magnetization = make_double3(temp.x, temp.y, normSq_s1 - normSq_s_1);

	size_t idx = dataZid * dimensions.x * dimensions.y * VALUES_IN_BLOCK + yid * dimensions.x * VALUES_IN_BLOCK + xid * VALUES_IN_BLOCK + dualNodeId;
	pMagnetization[idx] = dv * magnetization;
	density[idx] = dv * (normSq_s1 + normSq_s0 + normSq_s_1);
}

__global__ void integrate(double* dataVec, size_t stride, bool addLast)
{
	size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx >= stride)
	{
		return;
	}

	dataVec[idx] += dataVec[idx + stride];

	if ((idx == (stride - 1)) && addLast)
	{
		dataVec[idx] += dataVec[idx + stride + 1];
	}
}

__global__ void integrateVec(double3* dataVec, size_t stride, bool addLast)
{
	size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx >= stride)
	{
		return;
	}

	dataVec[idx] += dataVec[idx + stride];

	if ((idx == (stride - 1)) && addLast)
	{
		dataVec[idx] += dataVec[idx + stride + 1];
	}
}

__global__ void normalize(double* density, PitchedPtr psiPtr, uint3 dimensions)
{
	size_t xid = blockIdx.x * blockDim.x + threadIdx.x;
	size_t yid = blockIdx.y * blockDim.y + threadIdx.y;
	size_t zid = blockIdx.z * blockDim.z + threadIdx.z;
	size_t dataZid = zid / VALUES_IN_BLOCK; // One thread per every dual node so VALUES_IN_BLOCK threads per mesh block (on z-axis)

	// Exit leftover threads
	if (xid >= dimensions.x || yid >= dimensions.y || dataZid >= dimensions.z)
	{
		return;
	}

	size_t dualNodeId = zid % VALUES_IN_BLOCK; // Dual node id. One thread per every dual node so VALUES_IN_BLOCK threads per mesh block (on z-axis)
	BlockPsis* blockPsis = (BlockPsis*)(psiPtr.ptr + psiPtr.slicePitch * dataZid + psiPtr.pitch * yid) + xid;
	Complex3Vec psi = blockPsis->values[dualNodeId];
	double sqrtDens = sqrt(density[0]);
	psi.s1 = psi.s1 / sqrtDens;
	psi.s0 = psi.s0 / sqrtDens;
	psi.s_1 = psi.s_1 / sqrtDens;

	blockPsis->values[dualNodeId] = psi;
}

__global__ void polarState(PitchedPtr psi, uint3 dimensions)
{
	size_t xid = blockIdx.x * blockDim.x + threadIdx.x;
	size_t yid = blockIdx.y * blockDim.y + threadIdx.y;
	size_t zid = blockIdx.z * blockDim.z + threadIdx.z;

	size_t dataZid = zid / VALUES_IN_BLOCK; // One thread per every dual node so VALUES_IN_BLOCK threads per mesh block (on z-axis)

	// Exit leftover threads
	if (xid >= dimensions.x || yid >= dimensions.y || dataZid >= dimensions.z)
	{
		return;
	}

	BlockPsis* pPsi = (BlockPsis*)(psi.ptr + psi.slicePitch * dataZid + psi.pitch * yid) + xid;

	// Update psi
	size_t dualNodeId = zid % VALUES_IN_BLOCK; // Dual node id. One thread per every dual node so VALUES_IN_BLOCK threads per mesh block (on z-axis)

	Complex3Vec prev = pPsi->values[dualNodeId];

	double normSq_s1 = prev.s1.x * prev.s1.x + prev.s1.y * prev.s1.y;
	double normSq_s0 = prev.s0.x * prev.s0.x + prev.s0.y * prev.s0.y;
	double normSq_s_1 = prev.s_1.x * prev.s_1.x + prev.s_1.y * prev.s_1.y;
	double normSq = normSq_s1 + normSq_s0 + normSq_s_1;

	pPsi->values[dualNodeId].s1 = make_double2(0, 0);
	pPsi->values[dualNodeId].s0 = sqrt(normSq) * prev.s0 / sqrt(normSq_s0);
	pPsi->values[dualNodeId].s_1 = make_double2(0, 0);
};

__global__ void itp(PitchedPtr nextStep, PitchedPtr prevStep, int2* __restrict__ lapInd, double* __restrict__ hodges, uint3 dimensions, double block_scale, double3 p0, double c0, double c2, double alpha)
{
	size_t xid = blockIdx.x * blockDim.x + threadIdx.x;
	size_t yid = blockIdx.y * blockDim.y + threadIdx.y;
	size_t zid = blockIdx.z * blockDim.z + threadIdx.z;

	size_t dataZid = zid / VALUES_IN_BLOCK; // One thread per every dual node so VALUES_IN_BLOCK threads per mesh block (on z-axis)

	// Exit leftover threads
	if (xid >= dimensions.x || yid >= dimensions.y || dataZid >= dimensions.z)
	{
		return;
	}

	// Calculate the pointers for this block
	char* prevPsi = prevStep.ptr + prevStep.slicePitch * dataZid + prevStep.pitch * yid + sizeof(BlockPsis) * xid;
	BlockPsis* nextPsi = (BlockPsis*)(nextStep.ptr + nextStep.slicePitch * dataZid + nextStep.pitch * yid) + xid;

	// Update psi
	size_t dualNodeId = zid % VALUES_IN_BLOCK; // Dual node id. One thread per every dual node so VALUES_IN_BLOCK threads per mesh block (on z-axis)
	Complex3Vec prev = ((BlockPsis*)prevPsi)->values[dualNodeId];

	uint primaryFace = dualNodeId * FACE_COUNT;

	Complex3Vec H;
	H.s1 = make_double2(0, 0);
	H.s0 = make_double2(0, 0);
	H.s_1 = make_double2(0, 0);

	// Add the Laplacian to the Hamiltonian
#pragma unroll
	for (int i = 0; i < FACE_COUNT; ++i)
	{
		Complex3Vec otherBoundaryZeroCell = ((BlockPsis*)(prevPsi + lapInd[primaryFace].x))->values[lapInd[primaryFace].y];
		H.s1 += hodges[primaryFace] * (otherBoundaryZeroCell.s1 - prev.s1);
		H.s0 += hodges[primaryFace] * (otherBoundaryZeroCell.s0 - prev.s0);
		H.s_1 += hodges[primaryFace] * (otherBoundaryZeroCell.s_1 - prev.s_1);

		primaryFace++;
	}

	double normSq_s1 = prev.s1.x * prev.s1.x + prev.s1.y * prev.s1.y;
	double normSq_s0 = prev.s0.x * prev.s0.x + prev.s0.y * prev.s0.y;
	double normSq_s_1 = prev.s_1.x * prev.s_1.x + prev.s_1.y * prev.s_1.y;
	double normSq = normSq_s1 + normSq_s0 + normSq_s_1;

	double3 localPos = getLocalPos(dualNodeId);
	double3 globalPos = make_double3(p0.x + block_scale * (xid * BLOCK_WIDTH_X + localPos.x),
		p0.y + block_scale * (yid * BLOCK_WIDTH_Y + localPos.y),
		p0.z + block_scale * (dataZid * BLOCK_WIDTH_Z + localPos.z));
	double2 totalPot = make_double2(trap(globalPos) + c0 * normSq, -alpha * normSq * normSq);

	H.s1 += totalPot * prev.s1;
	H.s0 += totalPot * prev.s0;
	H.s_1 += totalPot * prev.s_1;

	double2 temp = SQRT_2 * (conj(prev.s1) * prev.s0 + conj(prev.s0) * prev.s_1);
	double3 magnetization = make_double3(temp.x, temp.y, normSq_s1 - normSq_s_1);
	double3 B = make_double3(0, 0, 0);
	B += c2 * magnetization;

	double2 Bxy = INV_SQRT_2 * make_double2(B.x, B.y);
	double2 BxyConj = conj(Bxy);

	H.s1 += (B.z * prev.s1 + BxyConj * prev.s0);
	H.s0 += (Bxy * prev.s1 + BxyConj * prev.s_1);
	H.s_1 += (Bxy * prev.s0 - B.z * prev.s_1);

	nextPsi->values[dualNodeId].s1 = prev.s1 - dt * make_double2(H.s1.x, H.s1.y);
	nextPsi->values[dualNodeId].s0 = prev.s0 - dt * make_double2(H.s0.x, H.s0.y);
	nextPsi->values[dualNodeId].s_1 = prev.s_1 - dt * make_double2(H.s_1.x, H.s_1.y);
};

__global__ void update(PitchedPtr nextStep, PitchedPtr prevStep, int2* __restrict__ lapInd, double* __restrict__ hodges, MagFields Bs, uint3 dimensions, double block_scale, double3 p0, double c0, double c2, double alpha)
{
	size_t xid = blockIdx.x * blockDim.x + threadIdx.x;
	size_t yid = blockIdx.y * blockDim.y + threadIdx.y;
	size_t zid = blockIdx.z * blockDim.z + threadIdx.z;

	size_t dataZid = zid / VALUES_IN_BLOCK; // One thread per every dual node so VALUES_IN_BLOCK threads per mesh block (on z-axis)

	// Exit leftover threads
	if (xid >= dimensions.x || yid >= dimensions.y || dataZid >= dimensions.z)
	{
		return;
	}

	// Calculate the pointers for this block
	char* prevPsi = prevStep.ptr + prevStep.slicePitch * dataZid + prevStep.pitch * yid + sizeof(BlockPsis) * xid;
	BlockPsis* nextPsi = (BlockPsis*)(nextStep.ptr + nextStep.slicePitch * dataZid + nextStep.pitch * yid) + xid;

	// Update psi
	size_t dualNodeId = zid % VALUES_IN_BLOCK; // Dual node id. One thread per every dual node so VALUES_IN_BLOCK threads per mesh block (on z-axis)
	Complex3Vec prev = ((BlockPsis*)prevPsi)->values[dualNodeId];

	uint primaryFace = dualNodeId * FACE_COUNT;

	Complex3Vec H;
	H.s1 = make_double2(0, 0);
	H.s0 = make_double2(0, 0);
	H.s_1 = make_double2(0, 0);

	// Add the Laplacian to the Hamiltonian
#pragma unroll
	for (int i = 0; i < FACE_COUNT; ++i)
	{
		Complex3Vec otherBoundaryZeroCell = ((BlockPsis*)(prevPsi + lapInd[primaryFace].x))->values[lapInd[primaryFace].y];
		H.s1 += hodges[primaryFace] * (otherBoundaryZeroCell.s1 - prev.s1);
		H.s0 += hodges[primaryFace] * (otherBoundaryZeroCell.s0 - prev.s0);
		H.s_1 += hodges[primaryFace] * (otherBoundaryZeroCell.s_1 - prev.s_1);

		primaryFace++;
	}

	double normSq_s1 = prev.s1.x * prev.s1.x + prev.s1.y * prev.s1.y;
	double normSq_s0 = prev.s0.x * prev.s0.x + prev.s0.y * prev.s0.y;
	double normSq_s_1 = prev.s_1.x * prev.s_1.x + prev.s_1.y * prev.s_1.y;
	double normSq = normSq_s1 + normSq_s0 + normSq_s_1;

	double3 localPos = getLocalPos(dualNodeId);
	double3 globalPos = make_double3(p0.x + block_scale * (xid * BLOCK_WIDTH_X + localPos.x),
		p0.y + block_scale * (yid * BLOCK_WIDTH_Y + localPos.y),
		p0.z + block_scale * (dataZid * BLOCK_WIDTH_Z + localPos.z));
	double2 totalPot = make_double2(trap(globalPos) + c0 * normSq, -alpha * normSq * normSq);

	H.s1 += totalPot * prev.s1;
	H.s0 += totalPot * prev.s0;
	H.s_1 += totalPot * prev.s_1;

	double2 temp = SQRT_2 * (conj(prev.s1) * prev.s0 + conj(prev.s0) * prev.s_1);
	double3 magnetization = make_double3(temp.x, temp.y, normSq_s1 - normSq_s_1);
	double3 B = magneticField(globalPos, Bs.Bq, Bs.Bz);
	B += c2 * magnetization;

	double2 Bxy = INV_SQRT_2 * make_double2(B.x, B.y);
	double2 BxyConj = conj(Bxy);

	// Linear Zeeman shift
	H.s1 += (B.z * prev.s1 + BxyConj * prev.s0);
	H.s0 += (Bxy * prev.s1 + BxyConj * prev.s_1);
	H.s_1 += (Bxy * prev.s0 - B.z * prev.s_1);

	// Quadratic Zeeman term
	B = magneticField(globalPos, Bs.BqQuad, Bs.BzQuad);
	Bxy = INV_SQRT_2 * make_double2(B.x, B.y);
	BxyConj = conj(Bxy);
	double BxyNormSq = (BxyConj * Bxy).x;
	double2 BxySq = Bxy * Bxy;
	double2 BxyConjSq = BxyConj * BxyConj;
	double BzSq = B.z * B.z;
	double2 BzBxy = B.z * Bxy;
	double2 BzBxyConj = B.z * BxyConj;

	H.s1 += (BzSq + BxyNormSq) * prev.s1 + BzBxyConj * prev.s0 + BxyConjSq * prev.s_1;
	H.s0 += BzBxy * prev.s1 + 2 * BxyNormSq * prev.s0 - BzBxyConj * prev.s_1;
	H.s_1 += BxySq * prev.s1 - BzBxy * prev.s0 + (BzSq + BxyNormSq) * prev.s_1;

	nextPsi->values[dualNodeId].s1 += dt * make_double2(H.s1.y, -H.s1.x);
	nextPsi->values[dualNodeId].s0 += dt * make_double2(H.s0.y, -H.s0.x);
	nextPsi->values[dualNodeId].s_1 += dt * make_double2(H.s_1.y, -H.s_1.x);
};

//void energy_h(dim3 dimGrid, dim3 dimBlock, double* energyPtr, PitchedPtr psi, PitchedPtr potentials, int2* lapInd, double* hodges, double g, uint3 dimensions, double volume, size_t bodies)
//{
//	energy << <dimGrid, dimBlock >> > (energyPtr, psi, potentials, lapInd, hodges, g, dimensions, volume);
//	int prevStride = bodies;
//	while (prevStride > 1)
//	{
//		int newStride = prevStride / 2;
//		integrate << <dim3(std::ceil(newStride / 32.0), 1, 1), dim3(32, 1, 1) >> > (energyPtr, newStride, ((newStride * 2) != prevStride));
//		prevStride = newStride;
//	}
//}

void normalize_h(dim3 dimGrid, dim3 dimBlock, double* densityPtr, PitchedPtr psi, uint3 dimensions, size_t bodies, double volume)
{
	density << <dimGrid, dimBlock >> > (densityPtr, psi, dimensions, volume);
	int prevStride = bodies;
	while (prevStride > 1)
	{
		int newStride = prevStride / 2;
		integrate << <dim3(std::ceil(newStride / 32.0), 1, 1), dim3(32, 1, 1) >> > (densityPtr, newStride, ((newStride * 2) != prevStride));
		prevStride = newStride;
	}

	normalize << < dimGrid, dimBlock >> > (densityPtr, psi, dimensions);
}

void printDensity(dim3 dimGrid, dim3 dimBlock, double* densityPtr, PitchedPtr psi, uint3 dimensions, size_t bodies, double volume)
{
	density << <dimGrid, dimBlock >> > (densityPtr, psi, dimensions, volume);
	int prevStride = bodies;
	while (prevStride > 1)
	{
		int newStride = prevStride / 2;
		integrate << <dim3(std::ceil(newStride / 32.0), 1, 1), dim3(32, 1, 1) >> > (densityPtr, newStride, ((newStride * 2) != prevStride));
		prevStride = newStride;
	}
	double hDensity = 0;
	checkCudaErrors(cudaMemcpy(&hDensity, densityPtr, sizeof(double), cudaMemcpyDeviceToHost));

	std::cout << "Total density: " << hDensity << std::endl;
}

double4 getMagnetizationAndDensity(dim3 dimGrid, dim3 dimBlock, double3* magnetizationPtr, double* densityPtr, PitchedPtr psi, uint3 dimensions, size_t bodies, double volume)
{
	magnetizationAndDensity << <dimGrid, dimBlock >> > (magnetizationPtr, densityPtr, psi, dimensions, volume);
	int prevStride = bodies;
	while (prevStride > 1)
	{
		int newStride = prevStride / 2;
		integrateVec << <dim3(std::ceil(newStride / 32.0), 1, 1), dim3(32, 1, 1) >> > (magnetizationPtr, newStride, ((newStride * 2) != prevStride));
		integrate << <dim3(std::ceil(newStride / 32.0), 1, 1), dim3(32, 1, 1) >> > (densityPtr, newStride, ((newStride * 2) != prevStride));
		prevStride = newStride;
	}
	double3 hMagnetization = make_double3(0, 0, 0);
	checkCudaErrors(cudaMemcpy(&hMagnetization, magnetizationPtr, sizeof(double3), cudaMemcpyDeviceToHost));

	double hDensity = 0;
	checkCudaErrors(cudaMemcpy(&hDensity, densityPtr, sizeof(double), cudaMemcpyDeviceToHost));

	return make_double4(hMagnetization.x, hMagnetization.y, hMagnetization.z, hDensity);
}

uint integrateInTime(const double block_scale, const Vector3& minp, const Vector3& maxp, const uint number_of_iterations)
{
	uint i, j, k, l;

	// find dimensions
	const Vector3 domain = maxp - minp;
	const uint xsize = uint(domain.x / (block_scale * BLOCK_WIDTH.x)) + 1;
	const uint ysize = uint(domain.y / (block_scale * BLOCK_WIDTH.y)) + 1;
	const uint zsize = uint(domain.z / (block_scale * BLOCK_WIDTH.z)) + 1;
	const Vector3 p0 = 0.5 * (minp + maxp - block_scale * Vector3(BLOCK_WIDTH.x * xsize, BLOCK_WIDTH.y * ysize, BLOCK_WIDTH.z * zsize));
	const double3 d_p0 = make_double3(p0.x, p0.y, p0.z);

	std::cout << xsize << ", " << ysize << ", " << zsize << std::endl;

	// compute discrete dimensions
	const uint bsize = VALUES_IN_BLOCK; // bpos.size(); // number of values inside a block
	const uint bxsize = (xsize + 1) * bsize; // number of values on x-row
	const uint bxysize = (ysize + 1) * bxsize; // number of values on xy-plane
	const uint ii0 = bxysize + bxsize + bsize; // reserved zeros in the beginning of value table
	const uint vsize = ii0 + (zsize + 1) * bxysize; // total number of values

	std::cout << "bsize: " << bsize << ", xsize: " << xsize << ", yszie: " << ysize << ", zsize: " << zsize << std::endl;
	uint64_t bodies = xsize * ysize * zsize * bsize;
	std::cout << "bodies = " << bodies << std::endl;

	// Initialize device memory
	size_t dxsize = xsize + 2; // One element buffer to both ends
	size_t dysize = ysize + 2; // One element buffer to both ends
	size_t dzsize = zsize + 2; // One element buffer to both ends
	cudaExtent psiExtent = make_cudaExtent(dxsize * sizeof(BlockPsis), dysize, dzsize);

	cudaPitchedPtr d_cudaEvenPsi;
	cudaPitchedPtr d_cudaOddPsi;
	checkCudaErrors(cudaMalloc3D(&d_cudaEvenPsi, psiExtent));
	checkCudaErrors(cudaMalloc3D(&d_cudaOddPsi, psiExtent));

	//double* d_energy;
	double* d_density;
	double3* d_magnetization;
	//checkCudaErrors(cudaMalloc(&d_energy, bodies * sizeof(double)));
	checkCudaErrors(cudaMalloc(&d_density, bodies * sizeof(double)));
	checkCudaErrors(cudaMalloc(&d_magnetization, bodies * sizeof(double3)));

	size_t offset = d_cudaEvenPsi.pitch * dysize + d_cudaEvenPsi.pitch + sizeof(BlockPsis);
	PitchedPtr d_evenPsi = { (char*)d_cudaEvenPsi.ptr + offset, d_cudaEvenPsi.pitch, d_cudaEvenPsi.pitch * dysize };
	PitchedPtr d_oddPsi = { (char*)d_cudaOddPsi.ptr + offset, d_cudaOddPsi.pitch, d_cudaOddPsi.pitch * dysize };

	// find terms for laplacian
	Buffer<int2> lapind;
	Buffer<double> hodges;
	double lapfac = -2.0 * 0.5 * getLaplacian(lapind, hodges, sizeof(BlockPsis), d_evenPsi.pitch, d_evenPsi.slicePitch) / (block_scale * block_scale);
	const uint lapsize = lapind.size() / bsize;
	double lapfac0 = lapsize * (-lapfac);

	//std::cout << "lapsize = " << lapsize << ", lapfac = " << lapfac << ", lapfac0 = " << lapfac0 << std::endl;

	// compute time step size
	const uint steps_per_iteration = 100; // 1.0 / 0.000199999994947575; // uint(iteration_period * (maxpot + lapfac0)) + 1; // number of time steps per iteration period

	std::cout << "steps_per_iteration = " << steps_per_iteration << std::endl;

	std::cout << "ALU operations per unit time = " << xsize * ysize * zsize * bsize * steps_per_iteration * FACE_COUNT << std::endl;

	for (int i = 0; i < hodges.size(); ++i) hodges[i] = -2.0 * 0.5 * hodges[i] / (block_scale * block_scale);

	int2* d_lapind;
	checkCudaErrors(cudaMalloc(&d_lapind, lapind.size() * sizeof(int2)));

	double* d_hodges;
	checkCudaErrors(cudaMalloc(&d_hodges, hodges.size() * sizeof(double)));

	// Initialize host memory
	size_t hostSize = dxsize * dysize * (zsize + 2);
	BlockPsis* h_evenPsi;// = new BlockPsis[dxsize * dysize * (zsize + 2)];
	BlockPsis* h_oddPsi;// = new BlockPsis[dxsize * dysize * (zsize + 2)];
	checkCudaErrors(cudaMallocHost(&h_evenPsi, hostSize * sizeof(BlockPsis)));
	checkCudaErrors(cudaMallocHost(&h_oddPsi, hostSize * sizeof(BlockPsis)));
	memset(h_evenPsi, 0, hostSize * sizeof(BlockPsis));
	memset(h_oddPsi, 0, hostSize * sizeof(BlockPsis));

#if COMPUTE_GROUND_STATE
	// Initialize discrete field
	Random rnd(54363);
	for (k = 0; k < zsize; k++)
	{
		for (j = 0; j < ysize; j++)
		{
			for (i = 0; i < xsize; i++)
			{
				for (l = 0; l < bsize; l++)
				{
					const uint srcI = ii0 + k * bxysize + j * bxsize + i * bsize + l;
					const uint dstI = (k + 1) * dxsize * dysize + (j + 1) * dxsize + (i + 1);
					const Vector2 s1 = rnd.getUniformCircle();
					const Vector2 s0 = rnd.getUniformCircle();
					const Vector2 s_1 = rnd.getUniformCircle();
					h_evenPsi[dstI].values[l].s1 = make_double2(s1.x, s1.y);
					h_evenPsi[dstI].values[l].s0 = make_double2(s0.x, s0.y);
					h_evenPsi[dstI].values[l].s_1 = make_double2(s_1.x, s_1.y);
				}
			}
		}
	}
#else
	std::ifstream fs(STATE_FILENAME, std::ios::binary | std::ios::in);
	if (fs.fail() != 0)
	{
		std::cout << "Failed to open file " << STATE_FILENAME << std::endl;
		return 1;
	}
	fs.read((char*)&h_evenPsi[0], hostSize * sizeof(BlockPsis));
	memcpy(&h_oddPsi[0], &h_evenPsi[0], hostSize * sizeof(BlockPsis));
	fs.close();
#endif

	cudaPitchedPtr h_cudaEvenPsi = { 0 };
	cudaPitchedPtr h_cudaOddPsi = { 0 };

	h_cudaEvenPsi.ptr = h_evenPsi;
	h_cudaEvenPsi.pitch = dxsize * sizeof(BlockPsis);
	h_cudaEvenPsi.xsize = d_cudaEvenPsi.xsize;
	h_cudaEvenPsi.ysize = d_cudaEvenPsi.ysize;

	h_cudaOddPsi.ptr = h_oddPsi;
	h_cudaOddPsi.pitch = dxsize * sizeof(BlockPsis);
	h_cudaOddPsi.xsize = d_cudaOddPsi.xsize;
	h_cudaOddPsi.ysize = d_cudaOddPsi.ysize;

	// Copy from host memory to device memory
	cudaMemcpy3DParms evenPsiParams = { 0 };
	cudaMemcpy3DParms oddPsiParams = { 0 };

	evenPsiParams.srcPtr = h_cudaEvenPsi;
	evenPsiParams.dstPtr = d_cudaEvenPsi;
	evenPsiParams.extent = psiExtent;
	evenPsiParams.kind = cudaMemcpyHostToDevice;

	oddPsiParams.srcPtr = h_cudaOddPsi;
	oddPsiParams.dstPtr = d_cudaOddPsi;
	oddPsiParams.extent = psiExtent;
	oddPsiParams.kind = cudaMemcpyHostToDevice;

	checkCudaErrors(cudaMemcpy3D(&evenPsiParams));
	checkCudaErrors(cudaMemcpy3D(&oddPsiParams));
	checkCudaErrors(cudaMemcpy(d_lapind, &lapind[0], lapind.size() * sizeof(int2), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_hodges, &hodges[0], hodges.size() * sizeof(double), cudaMemcpyHostToDevice));

	// Clear host memory after data has been copied to devices
	cudaDeviceSynchronize();
	lapind.clear();
	hodges.clear();
	cudaFreeHost(h_oddPsi);
#if !(SAVE_PICTURE || SAVE_VOLUME)
	cudaFreeHost(h_evenPsi);
#endif

	// Integrate in time
	uint3 dimensions = make_uint3(xsize, ysize, zsize);
	uint iter = 0;
	dim3 dimBlock(THREAD_BLOCK_X, THREAD_BLOCK_Y, THREAD_BLOCK_Z);
	dim3 dimGrid((xsize + THREAD_BLOCK_X - 1) / THREAD_BLOCK_X,
		(ysize + THREAD_BLOCK_Y - 1) / THREAD_BLOCK_Y,
		((zsize + THREAD_BLOCK_Z - 1) / THREAD_BLOCK_Z) * VALUES_IN_BLOCK);

	cudaMemcpy3DParms evenPsiBackParams = { 0 };
	evenPsiBackParams.srcPtr = d_cudaEvenPsi;
	evenPsiBackParams.dstPtr = h_cudaEvenPsi;
	evenPsiBackParams.extent = psiExtent;
	evenPsiBackParams.kind = cudaMemcpyDeviceToHost;

	const double volume = block_scale * block_scale * block_scale * VOLUME;
	double t = 0; // ms

#if COMPUTE_GROUND_STATE
	normalize_h(dimGrid, dimBlock, d_density, d_evenPsi, dimensions, bodies, volume);

	double mu = 0;
	double E = 1e20;
	auto Hpsi = d_oddPsi;
	while (true)
	{
		if ((iter % 1000) == 0) std::cout << "Iteration " << iter << std::endl;
#if SAVE_PICTURE
		if ((iter % SAVE_FREQUENCY) == 0)
		{
			checkCudaErrors(cudaMemcpy3D(&evenPsiBackParams));
			drawPicture("GS", h_evenPsi, dxsize, dysize, dzsize, iter, 0, 0, block_scale, d_p0);
			printDensity(dimGrid, dimBlock, d_density, d_evenPsi, dimensions, bodies, volume);
		}
#endif
		if (iter == 30000)
		{
			polarState << <dimGrid, dimBlock >> > (d_evenPsi, dimensions);
			checkCudaErrors(cudaMemcpy3D(&evenPsiBackParams));
			std::ofstream fs(STATE_FILENAME, std::ios::binary | std::ios_base::trunc);
			if (fs.fail() != 0) return 1;
			fs.write((char*)&h_evenPsi[0], hostSize * sizeof(BlockPsis));
			fs.close();
			return 0;
		}
		// Take an imaginary time step
		itp << <dimGrid, dimBlock >> > (d_oddPsi, d_evenPsi, d_lapind, d_hodges, dimensions, block_scale, d_p0, c0, c2, alpha);
		// Normalize
		normalize_h(dimGrid, dimBlock, d_density, d_oddPsi, dimensions, bodies, volume);

		// Take an imaginary time step
		itp << <dimGrid, dimBlock >> > (d_evenPsi, d_oddPsi, d_lapind, d_hodges, dimensions, block_scale, d_p0, c0, c2, alpha);
		// Normalize
		normalize_h(dimGrid, dimBlock, d_density, d_evenPsi, dimensions, bodies, volume);

		//energy_h(dimGrid, dimBlock, d_energy, d_evenPsi, d_pot, d_lapind, d_hodges, g, dimensions, volume, bodies);
		//double hDensity = 0;
		//double hEnergy = 0;
		//checkCudaErrors(cudaMemcpy(&hDensity, d_density, sizeof(double), cudaMemcpyDeviceToHost));
		//checkCudaErrors(cudaMemcpy(&hEnergy, d_energy, sizeof(double), cudaMemcpyDeviceToHost));

		//double newMu = hEnergy / hDensity;
		//double newE = hEnergy;
		//
		//std::cout << "Total density: " << hDensity << ", Total energy: " << hEnergy << ", mu: " << newMu << std::endl;

		//if (std::abs(mu - newMu) < 1e-4) break;

		//mu = newMu;
		//E = newE;

		iter++;
	}

#else
	constexpr uint saveFreq = 2000;

	Signal signal;
	MagFields Bs{};
	std::string times;
	std::string bqString;
	std::string bzString;
	std::string magX;
	std::string magY;
	std::string magZ;
	std::string densityStr;
	times += "times = [0";
	bqString += "Bq = [0";
	bzString += "Bz = [0";
	densityStr += "norm = [0";
	magX += "mag_x = [0";
	magY += "mag_y = [0";
	magZ += "mag_z = [0";
	while (true)
	{
#if SAVE_PICTURE
		// Copy back from device memory to host memory
		checkCudaErrors(cudaMemcpy3D(&evenPsiBackParams));

		// Measure wall clock time
		static auto prevTime = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::high_resolution_clock::now() - prevTime;
		std::cout << "Simulation time: " << t << " ms. Real time from previous save: " << duration.count() * 1e-9 << " s." << std::endl;

		drawPicture("TI", h_evenPsi, dxsize, dysize, dzsize, t, 0, 0, block_scale, d_p0);
		double4 magDens = getMagnetizationAndDensity(dimGrid, dimBlock, d_magnetization, d_density, d_evenPsi, dimensions, bodies, volume);
		times += ", " + std::to_string(t);
		bqString += ", " + std::to_string(Bs.Bq);
		bzString += ", " + std::to_string(Bs.Bz);
		magX += ", " + std::to_string(magDens.x);
		magY += ", " + std::to_string(magDens.y);
		magZ += ", " + std::to_string(magDens.z);
		densityStr += ", " + std::to_string(magDens.w);

		if (t > 300.0)
		{
			times += "];";
			bqString += "];";
			bzString += "];";
			magX += "];";
			magY += "];";
			magZ += "];";
			densityStr += "];";

			Text textFile;
			textFile << times << std::endl;
			textFile << bqString << std::endl;
			textFile << bzString << std::endl;
			textFile << magX << std::endl;
			textFile << magY << std::endl;
			textFile << magZ << std::endl;
			textFile << densityStr << std::endl;
			textFile.save("outputs.m");
			return 0;
		}
#endif
#if SAVE_VOLUME
		saveVolue(h_evenPsi, dxsize, dysize, dzsize, iter);
#endif
		prevTime = std::chrono::high_resolution_clock::now();
		// integrate one iteration
		for (uint step = 0; step < saveFreq; step++)
		{
			// update odd values
			signal = getSignal(t);
			Bs.Bq = BqScale * signal.Bq;
			Bs.Bz = BzScale * signal.Bz;
			Bs.BqQuad = BqQuadScale * signal.Bq;
			Bs.BzQuad = BzQuadScale * signal.Bz;
			update << <dimGrid, dimBlock >> > (d_oddPsi, d_evenPsi, d_lapind, d_hodges, Bs, dimensions, block_scale, d_p0, c0, c2, alpha);
			t += dt / omega_r * 1e3; // [ms]

			// update even values
			signal = getSignal(t);
			Bs.Bq = BqScale * signal.Bq;
			Bs.Bz = BzScale * signal.Bz;
			Bs.BqQuad = BqQuadScale * signal.Bq;
			Bs.BzQuad = BzQuadScale * signal.Bz;
			update << <dimGrid, dimBlock >> > (d_evenPsi, d_oddPsi, d_lapind, d_hodges, Bs, dimensions, block_scale, d_p0, c0, c2, alpha);
			t += dt / omega_r * 1e3; // [ms]
		}
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

int main(int argc, char** argv)
{
	const int number_of_iterations = 1000000;
	const double block_scale = Lx / Nx;

	std::cout << "block_scale = " << block_scale << std::endl;
	std::cout << "dual edge length = " << DUAL_EDGE_LENGTH * block_scale << std::endl;

	// integrate in time using DEC
	auto domainMin = Vector3(-Lx * 0.5, -Ly * 0.5, -Lz * 0.5);
	auto domainMax = Vector3(Lx * 0.5, Ly * 0.5, Lz * 0.5);
	integrateInTime(block_scale, domainMin, domainMax, number_of_iterations);

	return 0;
}
