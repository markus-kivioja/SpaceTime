#include <cuda_runtime.h>
#include <helper_cuda.h>

#include "AliceRingRamps.h"

#include "VortexState.hpp"
#include <Output/Picture.hpp>
#include <Output/Text.hpp>
#include <Types/Complex.hpp>
#include <Mesh/DelaunayMesh.hpp>
#include <iostream>
#include <sstream>
#include <chrono>
#include <random>

#include <mesh.h>

#define RELATIVISTIC 0

#define USE_QUADRUPOLE_OFFSET 0
#define USE_INITIAL_NOISE 0
#define SAVE_STATES 0
#define COMPUTE_GROUND_STATE 0
#define SAVE_PICTURE 1

#define THREAD_BLOCK_X 16
#define THREAD_BLOCK_Y 2
#define THREAD_BLOCK_Z 1

constexpr double DOMAIN_SIZE_X = 24.0;
constexpr double DOMAIN_SIZE_Y = 24.0;
constexpr double DOMAIN_SIZE_Z = 24.0;

constexpr double REPLICABLE_STRUCTURE_COUNT_X = 254.0;
//constexpr double REPLICABLE_STRUCTURE_COUNT_Y = 254.0;
//constexpr double REPLICABLE_STRUCTURE_COUNT_Z = 254.0;

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

constexpr double gamma = 2.9e-30;
const double alpha = N * N * gamma * 1e-12 / (a_r * a_r * a_r * a_r * a_r * a_r * 2 * PI * trapFreq_r);

constexpr double muB = 9.27400968e-24; // [m^2 kg / s^2 T^-1] Bohr magneton

const double BqScale = -(0.5 * muB / (hbar * omega_r) * a_r) / 100.; // [cm/Gauss]
constexpr double BzScale = -(0.5 * muB / (hbar * omega_r)) / 10000.; // [1/Gauss]

constexpr double A_hfs = 10.11734130545215;
const double BqQuadScale = 100 * a_r * sqrt(0.25 * 1000 * (1.399624624 * 1.399624624) / (trapFreq_r * 2 * A_hfs)); //[cm/Gauss]
const double BzQuadScale = sqrt(0.25 * 1000 * (1.399624624 * 1.399624624) / (trapFreq_r * 2 * A_hfs)); //[1/Gauss]  \sqrt{g_q}

constexpr double SQRT_2 = 1.41421356237309;
constexpr double INV_SQRT_2 = 0.70710678118655;

const std::string GROUND_STATE_FILENAME = "polar_ground_state.dat";

constexpr uint SAVE_FREQUENCY = 10;

#if USE_INITIAL_NOISE
constexpr double NOISE_AMPLITUDE = 0.1;
#endif

constexpr double dt = 3e-4; // 1 x // Before the monopole creation ramp (0 - 200 ms)
//constexpr double dt = 2e-5; // 0.1 x // During and after the monopole creation ramp (200 ms - )

#if RELATIVISTIC
constexpr double sigma = 0.01;
constexpr double dt_per_sigma = dt / sigma;
#endif

double t = 0; // Start time in ms
constexpr double END_TIME = 350; // End time in ms

inline __device__ __inline__ double trap(double3 p)
{
	double x = p.x * lambda_x;
	double y = p.y * lambda_y;
	double z = p.z * lambda_z;
	return 0.5 * (x * x + y * y + z * z) + 100.0;
}

__constant__ double quadrupoleCenterX = -0.20590789;
__constant__ double quadrupoleCenterY = -0.48902826;
__constant__ double quadrupoleCenterZ = -0.27353409;

inline __device__ __inline__ double3 magneticField(double3 p, double Bq, double Bz)
{
#if USE_QUADRUPOLE_OFFSET
	return {
		Bq * (p.x - quadrupoleCenterX),
		Bq * (p.y - quadrupoleCenterY),
		-2 * Bq * (p.z - quadrupoleCenterZ) + Bz
	};
#else
	return { Bq * p.x, Bq * p.y, -2 * Bq * p.z + Bz };
#endif
}

#include <utils.h>

__global__ void maxHamilton(double* maxHamlPtr, PitchedPtr prevStep, MagFields Bs, uint3 dimensions, double block_scale, double3 p0, double c0, double c2, double alpha)
{
	const size_t xid = blockIdx.x * blockDim.x + threadIdx.x;
	const size_t yid = blockIdx.y * blockDim.y + threadIdx.y;
	const size_t zid = blockIdx.z * blockDim.z + threadIdx.z;
	const size_t dataXid = xid / VALUES_IN_BLOCK; // One thread per every dual node so VALUES_IN_BLOCK threads per mesh block (on x-axis)

	// Exit leftover threads
	if (dataXid >= dimensions.x || yid >= dimensions.y || zid >= dimensions.z)
	{
		return;
	}

	// Calculate the pointers for this block
	char* prevPsi = prevStep.ptr + prevStep.slicePitch * zid + prevStep.pitch * yid + sizeof(BlockPsis) * dataXid;

	// Update psi
	const size_t dualNodeId = xid % VALUES_IN_BLOCK; // Dual node id. One thread per every dual node so VALUES_IN_BLOCK threads per mesh block (on x-axis)
	const Complex3Vec prev = ((BlockPsis*)prevPsi)->values[dualNodeId];

	const double normSq_s1 = prev.s1.x * prev.s1.x + prev.s1.y * prev.s1.y;
	const double normSq_s0 = prev.s0.x * prev.s0.x + prev.s0.y * prev.s0.y;
	const double normSq_s_1 = prev.s_1.x * prev.s_1.x + prev.s_1.y * prev.s_1.y;
	const double normSq = normSq_s1 + normSq_s0 + normSq_s_1;

	const double3 localPos = d_localPos[dualNodeId];
	const double3 globalPos = { p0.x + block_scale * (dataXid * BLOCK_WIDTH_X + localPos.x),
		p0.y + block_scale * (yid * BLOCK_WIDTH_Y + localPos.y),
		p0.z + block_scale * (zid * BLOCK_WIDTH_Z + localPos.z) };
	const double totalPot = trap(globalPos) + c0 * normSq;

	double3 hamilton = { totalPot, totalPot, totalPot };

	const double2 temp = SQRT_2 * (conj(prev.s1) * prev.s0 + conj(prev.s0) * prev.s_1);
	const double3 magnetization = { temp.x, temp.y, normSq_s1 - normSq_s_1 };
	double3 B = magneticField(globalPos, Bs.Bq, Bs.Bz);
	B += c2 * magnetization;

	// Linear Zeeman shift
	hamilton.x += abs(INV_SQRT_2 * B.x);
	hamilton.y += abs(INV_SQRT_2 * B.y);
	hamilton.z += abs(B.z);

	size_t idx = zid * dimensions.x * dimensions.y * VALUES_IN_BLOCK + yid * dimensions.x * VALUES_IN_BLOCK + dataXid * VALUES_IN_BLOCK + dualNodeId;
	maxHamlPtr[idx] = max(hamilton.x, max(hamilton.y, hamilton.z));
};

__global__ void density(double* density, PitchedPtr prevStep, uint3 dimensions, double dv)
{
	size_t xid = blockIdx.x * blockDim.x + threadIdx.x;
	size_t yid = blockIdx.y * blockDim.y + threadIdx.y;
	size_t zid = blockIdx.z * blockDim.z + threadIdx.z;
	size_t dataXid = xid / VALUES_IN_BLOCK; // One thread per every dual node so VALUES_IN_BLOCK threads per mesh block (on x-axis)

	// Exit leftover threads
	if (dataXid >= dimensions.x || yid >= dimensions.y || zid >= dimensions.z)
	{
		return;
	}

	size_t dualNodeId = xid % VALUES_IN_BLOCK; // Dual node id. One thread per every dual node so VALUES_IN_BLOCK threads per mesh block (on x-axis)

	char* pPsi = prevStep.ptr + prevStep.slicePitch * zid + prevStep.pitch * yid + sizeof(BlockPsis) * dataXid;
	Complex3Vec psi = ((BlockPsis*)pPsi)->values[dualNodeId];

	size_t idx = VALUES_IN_BLOCK * (zid * dimensions.x * dimensions.y + yid * dimensions.x + dataXid) + dualNodeId;
	density[idx] = dv * ((psi.s1 * conj(psi.s1)).x + (psi.s0 * conj(psi.s0)).x + (psi.s_1 * conj(psi.s_1)).x);
}

__global__ void magnetizationAndDensity(double3* pMagnetization, double* density, PitchedPtr prevStep, uint3 dimensions, double dv)
{
	size_t xid = blockIdx.x * blockDim.x + threadIdx.x;
	size_t yid = blockIdx.y * blockDim.y + threadIdx.y;
	size_t zid = blockIdx.z * blockDim.z + threadIdx.z;
	size_t dataXid = xid / VALUES_IN_BLOCK; // One thread per every dual node so VALUES_IN_BLOCK threads per mesh block (on x-axis)

	// Exit leftover threads
	if (dataXid >= dimensions.x || yid >= dimensions.y || zid >= dimensions.z)
	{
		return;
	}

	size_t dualNodeId = xid % VALUES_IN_BLOCK; // Dual node id. One thread per every dual node so VALUES_IN_BLOCK threads per mesh block (on x-axis)

	char* pPsi = prevStep.ptr + prevStep.slicePitch * zid + prevStep.pitch * yid + sizeof(BlockPsis) * dataXid;
	Complex3Vec psi = ((BlockPsis*)pPsi)->values[dualNodeId];

	double normSq_s1 = psi.s1.x * psi.s1.x + psi.s1.y * psi.s1.y;
	double normSq_s0 = psi.s0.x * psi.s0.x + psi.s0.y * psi.s0.y;
	double normSq_s_1 = psi.s_1.x * psi.s_1.x + psi.s_1.y * psi.s_1.y;
	double2 temp = SQRT_2 * (conj(psi.s1) * psi.s0 + conj(psi.s0) * psi.s_1);
	double3 magnetization = { temp.x, temp.y, normSq_s1 - normSq_s_1 };

	size_t idx = VALUES_IN_BLOCK * (zid * dimensions.x * dimensions.y + yid * dimensions.x + dataXid) + dualNodeId;
	pMagnetization[idx] = dv * magnetization;
	density[idx] = dv * (normSq_s1 + normSq_s0 + normSq_s_1);
}

__global__ void uvTheta(double3* out_u, double3* out_v, double* outTheta, PitchedPtr psiPtr, uint3 dimensions)
{
	size_t xid = blockIdx.x * blockDim.x + threadIdx.x;
	size_t yid = blockIdx.y * blockDim.y + threadIdx.y;
	size_t zid = blockIdx.z * blockDim.z + threadIdx.z;
	size_t dataXid = xid / VALUES_IN_BLOCK; // One thread per every dual node so VALUES_IN_BLOCK threads per mesh block (on x-axis)

	// Exit leftover threads
	if (dataXid >= dimensions.x || yid >= dimensions.y || zid >= dimensions.z)
	{
		return;
	}

	size_t dualNodeId = xid % VALUES_IN_BLOCK; // Dual node id. One thread per every dual node so VALUES_IN_BLOCK threads per mesh block (on x-axis)

	char* pPsi = psiPtr.ptr + psiPtr.slicePitch * zid + psiPtr.pitch * yid + sizeof(BlockPsis) * dataXid;
	Complex3Vec psi = ((BlockPsis*)pPsi)->values[dualNodeId];

	// a = m + in
	double2 ax = (psi.s_1 - psi.s1) / SQRT_2;
	double2 ay = double2{ 0, -1 } *(psi.s_1 + psi.s1) / SQRT_2;
	double2 az = psi.s0;
	double3 m = double3{ ax.x, ay.x, az.x };
	double3 n = double3{ ax.y, ay.y, az.y };

	double m_dot_n = m.x * n.x + m.y * n.y + m.z * n.z;
	double mNormSqr = m.x * m.x + m.y * m.y + m.z * m.z;
	double nNormSqr = n.x * n.x + n.y * n.y + n.z * n.z;

	double theta = atan2(-2 * m_dot_n, mNormSqr - nNormSqr) / 2;

	double sinTheta = sin(theta);
	double cosTheta = cos(theta);
	double3 u = double3{ m.x * cosTheta - sinTheta * n.x, m.y * cosTheta - sinTheta * n.y, m.z * cosTheta - sinTheta * n.z };
	double3 v = double3{ m.x * sinTheta + cosTheta * n.x, m.y * sinTheta + cosTheta * n.y, m.z * sinTheta + cosTheta * n.z };
	double uNorm = sqrt(u.x * u.x + u.y * u.y + u.z * u.z);
	double vNorm = sqrt(v.x * v.x + v.y * v.y + v.z * v.z);

	size_t idx = VALUES_IN_BLOCK * (zid * dimensions.x * dimensions.y + yid * dimensions.x + dataXid) + dualNodeId;
	if (uNorm >= vNorm)
	{
		out_u[idx] = u;
		out_v[idx] = v;
	}
	else
	{
		out_u[idx] = v;
		out_v[idx] = u;
	}
	outTheta[idx] = theta;
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

__global__ void reduceMax(double* dataVec, size_t stride, bool addLast)
{
	size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx >= stride)
	{
		return;
	}

	dataVec[idx] = max(dataVec[idx], dataVec[idx + stride]);

	if ((idx == (stride - 1)) && addLast)
	{
		dataVec[idx] = max(dataVec[idx], dataVec[idx + stride + 1]);
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
	size_t dataXid = xid / VALUES_IN_BLOCK; // One thread per every dual node so VALUES_IN_BLOCK threads per mesh block (on x-axis)

	// Exit leftover threads
	if (dataXid >= dimensions.x || yid >= dimensions.y || zid >= dimensions.z)
	{
		return;
	}

	size_t dualNodeId = xid % VALUES_IN_BLOCK; // Dual node id. One thread per every dual node so VALUES_IN_BLOCK threads per mesh block (on x-axis)
	BlockPsis* blockPsis = (BlockPsis*)(psiPtr.ptr + psiPtr.slicePitch * zid + psiPtr.pitch * yid) + dataXid;
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
	size_t dataXid = xid / VALUES_IN_BLOCK; // One thread per every dual node so VALUES_IN_BLOCK threads per mesh block (on x-axis)

	// Exit leftover threads
	if (dataXid >= dimensions.x || yid >= dimensions.y || zid >= dimensions.z)
	{
		return;
	}

	BlockPsis* pPsi = (BlockPsis*)(psi.ptr + psi.slicePitch * zid + psi.pitch * yid) + dataXid;

	// Update psi
	size_t dualNodeId = xid % VALUES_IN_BLOCK; // Dual node id. One thread per every dual node so VALUES_IN_BLOCK threads per mesh block (on x-axis)

	Complex3Vec prev = pPsi->values[dualNodeId];

	double normSq_s1 = prev.s1.x * prev.s1.x + prev.s1.y * prev.s1.y;
	double normSq_s0 = prev.s0.x * prev.s0.x + prev.s0.y * prev.s0.y;
	double normSq_s_1 = prev.s_1.x * prev.s_1.x + prev.s_1.y * prev.s_1.y;
	double normSq = normSq_s1 + normSq_s0 + normSq_s_1;

	pPsi->values[dualNodeId].s1 = { 0, 0 };
	pPsi->values[dualNodeId].s0 = sqrt(normSq) * prev.s0 / sqrt(normSq_s0);
	pPsi->values[dualNodeId].s_1 = { 0, 0 };
};

#if COMPUTE_GROUND_STATE
__global__ void itp(PitchedPtr nextStep, PitchedPtr prevStep, int4* __restrict__ laplace, double* __restrict__ hodges, uint3 dimensions, double block_scale, double3 p0, double c0, double c2, double alpha)
{
	const size_t xid = blockIdx.x * blockDim.x + threadIdx.x;
	const size_t yid = blockIdx.y * blockDim.y + threadIdx.y;
	const size_t zid = blockIdx.z * blockDim.z + threadIdx.z;
	const size_t dataXid = xid / VALUES_IN_BLOCK; // One thread per every dual node so VALUES_IN_BLOCK threads per mesh block (on x-axis)

	// Exit leftover threads
	if (dataXid > dimensions.x || yid > dimensions.y || zid > dimensions.z)
	{
		return;
	}

	const size_t localDataXid = threadIdx.x / VALUES_IN_BLOCK;

	__shared__ BlockPsis ldsPrevPsis[THREAD_BLOCK_Z * THREAD_BLOCK_Y * THREAD_BLOCK_X];
	const size_t threadIdxInBlock = threadIdx.z * THREAD_BLOCK_Y * THREAD_BLOCK_X + threadIdx.y * THREAD_BLOCK_X + localDataXid;

	// Calculate the pointers for this block
	char* prevPsi = prevStep.ptr + prevStep.slicePitch * zid + prevStep.pitch * yid + sizeof(BlockPsis) * dataXid;
	BlockPsis* nextPsi = (BlockPsis*)(nextStep.ptr + nextStep.slicePitch * zid + nextStep.pitch * yid) + dataXid;

	// Update psi
	const size_t dualNodeId = xid % VALUES_IN_BLOCK; // Dual node id. One thread per every dual node so VALUES_IN_BLOCK threads per mesh block (on x-axis)
	const Complex3Vec prev = ((BlockPsis*)prevPsi)->values[dualNodeId];
	ldsPrevPsis[threadIdxInBlock].values[dualNodeId] = prev;

	// Kill also the leftover edge threads
	if (dataXid == dimensions.x || yid == dimensions.y || zid == dimensions.z)
	{
		return;
	}
	__syncthreads();

	uint primaryFace = dualNodeId * FACE_COUNT;

	Complex3Vec H;
	H.s1 = { 0, 0 };
	H.s0 = { 0, 0 };
	H.s_1 = { 0, 0 };

	// Add the Laplacian to the Hamiltonian
#pragma unroll
	for (int i = 0; i < FACE_COUNT; ++i)
	{
		const int4 laplacian = laplace[primaryFace];

		const int neighbourX = localDataXid + laplacian.x;
		const int neighbourY = threadIdx.y + laplacian.y;
		const int neighbourZ = threadIdx.z + laplacian.z;

		Complex3Vec otherBoundaryZeroCell;
		// Read from the local shared memory
		if ((0 <= neighbourX) && (neighbourX < THREAD_BLOCK_X) &&
			(0 <= neighbourY) && (neighbourY < THREAD_BLOCK_Y) &&
			(0 <= neighbourZ) && (neighbourZ < THREAD_BLOCK_Z))
		{
			const int neighbourIdx = neighbourZ * THREAD_BLOCK_Y * THREAD_BLOCK_X + neighbourY * THREAD_BLOCK_X + neighbourX;
			otherBoundaryZeroCell = ldsPrevPsis[neighbourIdx].values[laplacian.w];
		}
		else // Read from the global memory
		{
			const int offset = laplacian.z * prevStep.slicePitch + laplacian.y * prevStep.pitch + laplacian.x * sizeof(BlockPsis);
			otherBoundaryZeroCell = ((BlockPsis*)(prevPsi + offset))->values[laplacian.w];
		}

		const double hodge = hodges[primaryFace];
		H.s1 += hodge * (otherBoundaryZeroCell.s1 - prev.s1);
		H.s0 += hodge * (otherBoundaryZeroCell.s0 - prev.s0);
		H.s_1 += hodge * (otherBoundaryZeroCell.s_1 - prev.s_1);

		primaryFace++;
	}

	const double normSq_s1 = prev.s1.x * prev.s1.x + prev.s1.y * prev.s1.y;
	const double normSq_s0 = prev.s0.x * prev.s0.x + prev.s0.y * prev.s0.y;
	const double normSq_s_1 = prev.s_1.x * prev.s_1.x + prev.s_1.y * prev.s_1.y;
	const double normSq = normSq_s1 + normSq_s0 + normSq_s_1;

	const double3 localPos = getLocalPos(dualNodeId);
	const double3 globalPos = { p0.x + block_scale * (dataXid * BLOCK_WIDTH_X + localPos.x),
		p0.y + block_scale * (yid * BLOCK_WIDTH_Y + localPos.y),
		p0.z + block_scale * (zid * BLOCK_WIDTH_Z + localPos.z));
	const double totalPot = trap(globalPos) + c0 * normSq;

	H.s1 += totalPot * prev.s1;
	H.s0 += totalPot * prev.s0;
	H.s_1 += totalPot * prev.s_1;

	const double2 temp = SQRT_2 * (conj(prev.s1) * prev.s0 + conj(prev.s0) * prev.s_1);
	const double3 magnetization = { temp.x, temp.y, normSq_s1 - normSq_s_1);
	double3 B = c2 * magnetization;

	// Linear Zeeman shift
	double2 Bxy = INV_SQRT_2* { B.x, B.y);
	double2 BxyConj = conj(Bxy);
	H.s1 += (B.z * prev.s1 + BxyConj * prev.s0);
	H.s0 += (Bxy * prev.s1 + BxyConj * prev.s_1);
	H.s_1 += (Bxy * prev.s0 - B.z * prev.s_1);

	nextPsi->values[dualNodeId].s1 = prev.s1 - dt * double2{ H.s1.x, H.s1.y};
	nextPsi->values[dualNodeId].s0 = prev.s0 - dt * double2{ H.s0.x, H.s0.y};
	nextPsi->values[dualNodeId].s_1 = prev.s_1 - dt * double2{ H.s_1.x, H.s_1.y };
};
#else
__global__ void forwardEuler(PitchedPtr nextStep, PitchedPtr prevStep, PitchedPtr qs, const int2 * __restrict__ d1Ptr, const double* __restrict__ hodges, MagFields Bs, const uint3 dimensions, const double block_scale, const double3 p0, const double c0, const double c2)
{
	const size_t xid = blockIdx.x * blockDim.x + threadIdx.x;
	const size_t yid = blockIdx.y * blockDim.y + threadIdx.y;
	const size_t zid = blockIdx.z * blockDim.z + threadIdx.z;
	const size_t dataXid = xid / VALUES_IN_BLOCK; // One thread per every dual node so VALUES_IN_BLOCK threads per mesh block (on x-axis)
	const size_t dualNodeId = xid % VALUES_IN_BLOCK; // Dual node id. One thread per every dual node so VALUES_IN_BLOCK threads per mesh block (on x-axis)

	// Exit leftover threads
	if (dataXid >= dimensions.x || yid >= dimensions.y || zid >= dimensions.z)
	{
		return;
	}

	// Calculate the pointers for this block
	char* prevPsi = prevStep.ptr + prevStep.slicePitch * zid + prevStep.pitch * yid + sizeof(BlockPsis) * dataXid;
	char* qPtr = qs.ptr + qs.slicePitch * zid + qs.pitch * yid + sizeof(BlockEdges) * dataXid;
	BlockPsis* nextPsi = (BlockPsis*)(nextStep.ptr + nextStep.slicePitch * zid + nextStep.pitch * yid) + dataXid;

	// Update psi
	const Complex3Vec prev = ((BlockPsis*)prevPsi)->values[dualNodeId];

	Complex3Vec H;
	H.s1 = { 0, 0 };
	H.s0 = { 0, 0 };
	H.s_1 = { 0, 0 };

	// Add the Laplacian to the Hamiltonian
	// Add the Laplacian (d1 of d0) to the Hamiltonian
	uint startEdgeId = dualNodeId * FACE_COUNT;
#pragma unroll
	for (int edgeIdOffset = 0; edgeIdOffset < FACE_COUNT; ++edgeIdOffset)
	{
		int edgeId = startEdgeId + edgeIdOffset;
		int2 d1 = d1Ptr[edgeId];
		Complex3Vec d0psi = ((BlockEdges*)(qPtr + d1.x))->values[d1.y];
		const double hodge = hodges[edgeId];

		H.s1 += hodge * d0psi.s1;
		H.s0 += hodge * d0psi.s0;
		H.s_1 += hodge * d0psi.s_1;
	}
#if RELATIVISTIC
	H.s1 = -1.0 * H.s1;
	H.s0 = -1.0 * H.s0;
	H.s_1 = -1.0 * H.s_1;
#endif

	const double normSq_s1 = prev.s1.x * prev.s1.x + prev.s1.y * prev.s1.y;
	const double normSq_s0 = prev.s0.x * prev.s0.x + prev.s0.y * prev.s0.y;
	const double normSq_s_1 = prev.s_1.x * prev.s_1.x + prev.s_1.y * prev.s_1.y;
	const double normSq = normSq_s1 + normSq_s0 + normSq_s_1;

	const double3 localPos = d_localPos[dualNodeId];
	const double3 globalPos = { p0.x + block_scale * (dataXid * BLOCK_WIDTH_X + localPos.x),
		p0.y + block_scale * (yid * BLOCK_WIDTH_Y + localPos.y),
		p0.z + block_scale * (zid * BLOCK_WIDTH_Z + localPos.z) };
	//const double2 totalPot = {trap(globalPos) + c0 * normSq, -alpha * normSq * normSq};
	const double totalPot = trap(globalPos) + c0 * normSq;

	H.s1 += totalPot * prev.s1;
	H.s0 += totalPot * prev.s0;
	H.s_1 += totalPot * prev.s_1;

	const double2 temp = SQRT_2 * (conj(prev.s1) * prev.s0 + conj(prev.s0) * prev.s_1);
	const double3 magnetization = { temp.x, temp.y, normSq_s1 - normSq_s_1 };
	double3 B = magneticField(globalPos, Bs.Bq, Bs.Bz);
	B += c2 * magnetization;

	// Linear Zeeman shift
	double2 Bxy = INV_SQRT_2 * double2{ B.x, B.y };
	double2 BxyConj = conj(Bxy);
	H.s1 += (B.z * prev.s1 + BxyConj * prev.s0);
	H.s0 += (Bxy * prev.s1 + BxyConj * prev.s_1);
	H.s_1 += (Bxy * prev.s0 - B.z * prev.s_1);

	// Quadratic Zeeman term
	//B = magneticField(globalPos, Bs.BqQuad, Bs.BzQuad);
	//Bxy = INV_SQRT_2 * {B.x, B.y};
	//BxyConj = conj(Bxy);
	//double BxyNormSq = (BxyConj * Bxy).x;
	//double2 BxySq = Bxy * Bxy;
	//double2 BxyConjSq = BxyConj * BxyConj;
	//double BzSq = B.z * B.z;
	//double2 BzBxy = B.z * Bxy;
	//double2 BzBxyConj = B.z * BxyConj;
	//H.s1 += (BzSq + BxyNormSq) * prev.s1 + BzBxyConj * prev.s0 + BxyConjSq * prev.s_1;
	//H.s0 += BzBxy * prev.s1 + 2 * BxyNormSq * prev.s0 - BzBxyConj * prev.s_1;
	//H.s_1 += BxySq * prev.s1 - BzBxy * prev.s0 + (BzSq + BxyNormSq) * prev.s_1;

	nextPsi->values[dualNodeId].s1 = prev.s1 + 0.5 * dt * double2{ H.s1.y, -H.s1.x };
	nextPsi->values[dualNodeId].s0 = prev.s0 + 0.5 * dt * double2{ H.s0.y, -H.s0.x };
	nextPsi->values[dualNodeId].s_1 = prev.s_1 + 0.5 * dt * double2{ H.s_1.y, -H.s_1.x };
};

__global__ void update_q(PitchedPtr next_q, PitchedPtr prev_q, PitchedPtr psi, int3* d0, uint3 dimensions)
{
	size_t xid = blockIdx.x * blockDim.x + threadIdx.x;
	size_t yid = blockIdx.y * blockDim.y + threadIdx.y;
	size_t zid = blockIdx.z * blockDim.z + threadIdx.z;

	size_t dataXid = xid / EDGES_IN_BLOCK; // One thread per every dual edge so EDGES_IN_BLOCK threads per mesh block (on z-axis)
	size_t dualEdgeId = xid % EDGES_IN_BLOCK; // Dual node id. One thread per every dual node so VALUES_IN_BLOCK threads per mesh block (on z-axis)

	// Exit leftover threads
	if (dataXid >= dimensions.x || yid >= dimensions.y || zid >= dimensions.z)
	{
		return;
	}

	char* pPsi = psi.ptr + psi.slicePitch * zid + psi.pitch * yid + sizeof(BlockPsis) * dataXid;

	Complex3Vec thisPsi = ((BlockPsis*)(pPsi))->values[d0[dualEdgeId].x];
	Complex3Vec otherPsi = ((BlockPsis*)(pPsi + d0[dualEdgeId].y))->values[d0[dualEdgeId].z];
	Complex3Vec d0psi;
	d0psi.s1 = otherPsi.s1 - thisPsi.s1;
	d0psi.s0 = otherPsi.s0 - thisPsi.s0;
	d0psi.s_1 = otherPsi.s_1 - thisPsi.s_1;

	BlockEdges* next = (BlockEdges*)(next_q.ptr + next_q.slicePitch * zid + next_q.pitch * yid) + dataXid;
#if RELATIVISTIC
	BlockEdges* prev = (BlockEdges*)(prev_q.ptr + prev_q.slicePitch * zid + prev_q.pitch * yid) + dataXid;
	
	Complex3Vec q;
	q.s1 = dt_per_sigma * (prev->values[dualEdgeId].s1 + d0psi.s1);
	q.s0 = dt_per_sigma * (prev->values[dualEdgeId].s0 + d0psi.s0);
	q.s_1 = dt_per_sigma * (prev->values[dualEdgeId].s_1 + d0psi.s_1);

	next->values[dualEdgeId].s1 += make_double2(-q.s1.y, q.s1.x);
	next->values[dualEdgeId].s0 += make_double2(-q.s0.y, q.s0.x);
	next->values[dualEdgeId].s_1 += make_double2(-q.s_1.y, q.s_1.x);
#else
	next->values[dualEdgeId] = d0psi;
#endif
}

__global__ void update_psi(PitchedPtr nextStep, PitchedPtr prevStep, PitchedPtr qs, const int2* __restrict__ d1Ptr, const double* __restrict__ hodges, MagFields Bs, const uint3 dimensions, const double block_scale, const double3 p0, const double c0, const double c2)
{
	const size_t xid = blockIdx.x * blockDim.x + threadIdx.x;
	const size_t yid = blockIdx.y * blockDim.y + threadIdx.y;
	const size_t zid = blockIdx.z * blockDim.z + threadIdx.z;
	const size_t dataXid = xid / VALUES_IN_BLOCK; // One thread per every dual node so VALUES_IN_BLOCK threads per mesh block (on x-axis)
	const size_t dualNodeId = xid % VALUES_IN_BLOCK; // Dual node id. One thread per every dual node so VALUES_IN_BLOCK threads per mesh block (on x-axis)

	// Exit leftover threads
	if (dataXid >= dimensions.x || yid >= dimensions.y || zid >= dimensions.z)
	{
		return;
	}

	// Calculate the pointers for this block
	char* prevPsi = prevStep.ptr + prevStep.slicePitch * zid + prevStep.pitch * yid + sizeof(BlockPsis) * dataXid;
	char* qPtr = qs.ptr + qs.slicePitch * zid + qs.pitch * yid + sizeof(BlockEdges) * dataXid;
	BlockPsis* nextPsi = (BlockPsis*)(nextStep.ptr + nextStep.slicePitch * zid + nextStep.pitch * yid) + dataXid;

	// Update psi
	const Complex3Vec prev = ((BlockPsis*)prevPsi)->values[dualNodeId];

	Complex3Vec H;
	H.s1 = { 0, 0 };
	H.s0 = { 0, 0 };
	H.s_1 = { 0, 0 };

	// Add the Laplacian (d1 of d0) to the Hamiltonian
	uint startEdgeId = dualNodeId * FACE_COUNT;
#pragma unroll
	for (int edgeIdOffset = 0; edgeIdOffset < FACE_COUNT; ++edgeIdOffset)
	{
		int edgeId = startEdgeId + edgeIdOffset;
		int2 d1 = d1Ptr[edgeId];
		Complex3Vec d0psi = ((BlockEdges*)(qPtr + d1.x))->values[d1.y];
		const double hodge = hodges[edgeId];

		H.s1 += hodge * d0psi.s1;
		H.s0 += hodge * d0psi.s0;
		H.s_1 += hodge * d0psi.s_1;
	}
#if RELATIVISTIC
	H.s1 = -1.0 * H.s1;
	H.s0 = -1.0 * H.s0;
	H.s_1 = -1.0 * H.s_1;
#endif

	const double normSq_s1 = prev.s1.x * prev.s1.x + prev.s1.y * prev.s1.y;
	const double normSq_s_1 = prev.s_1.x * prev.s_1.x + prev.s_1.y * prev.s_1.y;
	const double normSq = normSq_s1 + (prev.s0.x * prev.s0.x + prev.s0.y * prev.s0.y) + normSq_s_1;

	const double3 localPos = d_localPos[dualNodeId];
	const double3 globalPos = { p0.x + block_scale * (dataXid * BLOCK_WIDTH_X + localPos.x),
		p0.y + block_scale * (yid * BLOCK_WIDTH_Y + localPos.y),
		p0.z + block_scale * (zid * BLOCK_WIDTH_Z + localPos.z) };
	const double totalPot = trap(globalPos) + c0 * normSq;

	H.s1 += totalPot * prev.s1;
	H.s0 += totalPot * prev.s0;
	H.s_1 += totalPot * prev.s_1;

	const double2 magXY = SQRT_2 * (conj(prev.s1) * prev.s0 + conj(prev.s0) * prev.s_1);
	double3 B = magneticField(globalPos, Bs.Bq, Bs.Bz);
	B += c2 * double3{ magXY.x, magXY.y, normSq_s1 - normSq_s_1 };

	// Linear Zeeman shift
	const double2 Bxy = INV_SQRT_2 * double2{ B.x, B.y };
	const double2 BxyConj = conj(Bxy);
	H.s1 += (B.z * prev.s1 + BxyConj * prev.s0);
	H.s0 += (Bxy * prev.s1 + BxyConj * prev.s_1);
	H.s_1 += (Bxy * prev.s0 - B.z * prev.s_1);

	nextPsi->values[dualNodeId].s1 += dt * double2{ H.s1.y, -H.s1.x };
	nextPsi->values[dualNodeId].s0 += dt * double2{ H.s0.y, -H.s0.x };
	nextPsi->values[dualNodeId].s_1 += dt * double2{ H.s_1.y, -H.s_1.x };
};
#endif
//void energy_h(dim3 dimGrid, dim3 dimBlock, double* energyPtr, PitchedPtr psi, PitchedPtr potentials, int4* lapInd, double* hodges, double g, uint3 dimensions, double volume, size_t bodies)
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
	double3 hMagnetization = { 0, 0, 0 };
	checkCudaErrors(cudaMemcpy(&hMagnetization, magnetizationPtr, sizeof(double3), cudaMemcpyDeviceToHost));

	double hDensity = 0;
	checkCudaErrors(cudaMemcpy(&hDensity, densityPtr, sizeof(double), cudaMemcpyDeviceToHost));

	return { hMagnetization.x, hMagnetization.y, hMagnetization.z, hDensity };
}

float getMaxHamilton(dim3 dimGrid, dim3 dimBlock, double* maxHamlPtr, PitchedPtr psi, MagFields Bs, uint3 dimensions, size_t bodies, double block_scale, double3 p0)
{
	maxHamilton << <dimGrid, dimBlock >> > (maxHamlPtr, psi, Bs, dimensions, block_scale, p0, c0, c2, alpha);
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

uint integrateInTime(const double block_scale, const Vector3& minp, const Vector3& maxp)
{
	// find dimensions
	const Vector3 domain = maxp - minp;
	const uint xsize = uint(domain.x / (block_scale * BLOCK_WIDTH.x)) + 1;
	const uint ysize = uint(domain.y / (block_scale * BLOCK_WIDTH.y)) + 1;
	const uint zsize = uint(domain.z / (block_scale * BLOCK_WIDTH.z)) + 1;
	const Vector3 p0 = 0.5 * (minp + maxp - block_scale * Vector3(BLOCK_WIDTH.x * xsize, BLOCK_WIDTH.y * ysize, BLOCK_WIDTH.z * zsize));
	const double3 d_p0 = { p0.x, p0.y, p0.z };

	// compute discrete dimensions
	const uint bsize = VALUES_IN_BLOCK; // bpos.size(); // number of values inside a block

	std::cout << "Dual 0-cells in a replicable structure: " << bsize << std::endl;
	std::cout << "Replicable structure instances in x: " << xsize << ", y: " << ysize << ", z: " << zsize << std::endl;
	uint64_t bodies = xsize * ysize * zsize * bsize;
	std::cout << "Dual 0-cells in total: " << bodies << std::endl;

	// Initialize device memory
	size_t dxsize = xsize + 2; // One element buffer to both ends
	size_t dysize = ysize + 2; // One element buffer to both ends
	size_t dzsize = zsize + 2; // One element buffer to both ends
	cudaExtent psiExtent = make_cudaExtent(dxsize * sizeof(BlockPsis), dysize, dzsize);
	cudaExtent edgeExtent = make_cudaExtent(dxsize * sizeof(BlockEdges), dysize, dzsize);

	cudaPitchedPtr d_cudaEvenPsi;
	cudaPitchedPtr d_cudaEvenQ;
	cudaPitchedPtr d_cudaOddPsi;
	cudaPitchedPtr d_cudaOddQ;
	checkCudaErrors(cudaMalloc3D(&d_cudaEvenPsi, psiExtent));
	checkCudaErrors(cudaMalloc3D(&d_cudaEvenQ, edgeExtent));
	checkCudaErrors(cudaMalloc3D(&d_cudaOddPsi, psiExtent));
	checkCudaErrors(cudaMalloc3D(&d_cudaOddQ, edgeExtent));

	//double* d_energy;
	double* d_density;
	double3* d_magnetization;
	double3* d_u;
	double3* d_v;
	double* d_theta;
	//checkCudaErrors(cudaMalloc(&d_energy, bodies * sizeof(double)));
	checkCudaErrors(cudaMalloc(&d_density, bodies * sizeof(double)));
	checkCudaErrors(cudaMalloc(&d_magnetization, bodies * sizeof(double3)));
	checkCudaErrors(cudaMalloc(&d_u, bodies * sizeof(double3)));
	checkCudaErrors(cudaMalloc(&d_v, bodies * sizeof(double3)));
	checkCudaErrors(cudaMalloc(&d_theta, bodies * sizeof(double)));

	size_t offset = d_cudaEvenPsi.pitch * dysize + d_cudaEvenPsi.pitch + sizeof(BlockPsis);
	size_t edgeOffset = d_cudaEvenQ.pitch * dysize + d_cudaEvenQ.pitch + sizeof(BlockEdges);
	PitchedPtr d_evenPsi = { (char*)d_cudaEvenPsi.ptr + offset, d_cudaEvenPsi.pitch, d_cudaEvenPsi.pitch * dysize };
	PitchedPtr d_evenQ = { (char*)d_cudaEvenQ.ptr + edgeOffset, d_cudaEvenQ.pitch, d_cudaEvenQ.pitch * dysize };
	PitchedPtr d_oddPsi = { (char*)d_cudaOddPsi.ptr + offset, d_cudaOddPsi.pitch, d_cudaOddPsi.pitch * dysize };
	PitchedPtr d_oddQ = { (char*)d_cudaOddQ.ptr + edgeOffset, d_cudaOddQ.pitch, d_cudaOddQ.pitch * dysize };

	// find terms for laplacian
	Buffer<int3> d0;
	Buffer<int2> d1;
	Buffer<double> hodges;
	double lapfac = -2.0 * 0.5 * getLaplacian(hodges, d0, d1, sizeof(BlockPsis), d_evenPsi.pitch, d_evenPsi.slicePitch, sizeof(BlockEdges), d_evenQ.pitch, d_evenQ.slicePitch) / (block_scale * block_scale);
	const uint lapsize = hodges.size() / bsize;
	double lapfac0 = lapsize * (-lapfac);

	//std::cout << "lapsize = " << lapsize << ", lapfac = " << lapfac << ", lapfac0 = " << lapfac0 << std::endl;

	std::cout << "ALU operations per iteration = " << xsize * ysize * zsize * bsize * FACE_COUNT << std::endl;

	for (int i = 0; i < hodges.size(); ++i) hodges[i] = -2.0 * 0.5 * hodges[i] / (block_scale * block_scale);

	int3* d_d0;
	checkCudaErrors(cudaMalloc(&d_d0, d0.size() * sizeof(int3)));

	int2* d_d1;
	checkCudaErrors(cudaMalloc(&d_d1, d1.size() * sizeof(int2)));

	double* d_hodges;
	checkCudaErrors(cudaMalloc(&d_hodges, hodges.size() * sizeof(double)));

	// Initialize host memory
	size_t hostSize = dxsize * dysize * (zsize + 2);
	BlockPsis* h_evenPsi;
	BlockPsis* h_oddPsi;
	checkCudaErrors(cudaMallocHost(&h_evenPsi, hostSize * sizeof(BlockPsis)));
	checkCudaErrors(cudaMallocHost(&h_oddPsi, hostSize * sizeof(BlockPsis)));
	memset(h_evenPsi, 0, hostSize * sizeof(BlockPsis));
	memset(h_oddPsi, 0, hostSize * sizeof(BlockPsis));

	double* h_density;
	double3* h_magnetization;
	double3* h_u;
	double3* h_v;
	double* h_theta;
	checkCudaErrors(cudaMallocHost(&h_density, bodies * sizeof(double)));
	checkCudaErrors(cudaMallocHost(&h_magnetization, bodies * sizeof(double3)));
	checkCudaErrors(cudaMallocHost(&h_u, bodies * sizeof(double3)));
	checkCudaErrors(cudaMallocHost(&h_v, bodies * sizeof(double3)));
	checkCudaErrors(cudaMallocHost(&h_theta, bodies * sizeof(double)));

#if COMPUTE_GROUND_STATE
	// Initialize discrete field
	Random rnd(54363);
	for (uint k = 0; k < zsize; k++)
	{
		for (uint j = 0; j < ysize; j++)
		{
			for (uint i = 0; i < xsize; i++)
			{
				for (uint l = 0; l < bsize; l++)
				{
					const uint dstI = (k + 1) * dxsize * dysize + (j + 1) * dxsize + (i + 1);
					const Vector2 s1 = rnd.getUniformCircle();
					const Vector2 s0 = rnd.getUniformCircle();
					const Vector2 s_1 = rnd.getUniformCircle();
					h_evenPsi[dstI].values[l].s1 = {s1.x, s1.y};
					h_evenPsi[dstI].values[l].s0 = {s0.x, s0.y};
					h_evenPsi[dstI].values[l].s_1 = { s_1.x, s_1.y };
				}
			}
		}
	}
#else
	bool loadGroundState = (t == 0);
	std::string filename = loadGroundState ? GROUND_STATE_FILENAME : toString(t) + ".dat";
	std::ifstream fs(filename, std::ios::binary | std::ios::in);
	if (fs.fail() != 0)
	{
		std::cout << "Failed to open file " << filename << std::endl;
		return 1;
	}
	fs.read((char*)&h_oddPsi[0], hostSize * sizeof(BlockPsis));
	fs.close();

#if USE_INITIAL_NOISE
	if (loadGroundState && (NOISE_AMPLITUDE > 0))
	{
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
						// Add noise
						const uint dstI = (k + 1) * dxsize * dysize + (j + 1) * dxsize + (i + 1);
						const double2 rand = { distribution(generator), distribution(generator) };
						const double dens = (conj(h_oddPsi[dstI].values[l].s0) * h_oddPsi[dstI].values[l].s0).x;
						h_oddPsi[dstI].values[l].s0 += sqrt(dens) * NOISE_AMPLITUDE * rand;

						// Normalize
						const double newDens = (conj(h_oddPsi[dstI].values[l].s0) * h_oddPsi[dstI].values[l].s0).x;
						h_oddPsi[dstI].values[l].s0 = sqrt(dens / newDens) * h_oddPsi[dstI].values[l].s0;
					}
				}
			}
		}
	}
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
	cudaMemcpy3DParms evenPsiBackParams = { 0 };
	evenPsiBackParams.srcPtr = d_cudaEvenPsi;
	evenPsiBackParams.dstPtr = h_cudaEvenPsi;
	evenPsiBackParams.extent = psiExtent;
	evenPsiBackParams.kind = cudaMemcpyDeviceToHost;

	cudaMemcpy3DParms oddPsiBackParams = { 0 };
	oddPsiBackParams.srcPtr = d_cudaOddPsi;
	oddPsiBackParams.dstPtr = h_cudaOddPsi;
	oddPsiBackParams.extent = psiExtent;
	oddPsiBackParams.kind = cudaMemcpyDeviceToHost;

	// Integrate in time
	uint3 dimensions = make_uint3(xsize, ysize, zsize);
	dim3 psiDimBlock(THREAD_BLOCK_X * VALUES_IN_BLOCK, THREAD_BLOCK_Y, THREAD_BLOCK_Z);
	dim3 edgeDimBlock(THREAD_BLOCK_X * EDGES_IN_BLOCK, THREAD_BLOCK_Y, THREAD_BLOCK_Z);
	dim3 dimGrid((xsize + THREAD_BLOCK_X - 1) / THREAD_BLOCK_X,
		(ysize + THREAD_BLOCK_Y - 1) / THREAD_BLOCK_Y,
		((zsize + THREAD_BLOCK_Z - 1) / THREAD_BLOCK_Z));

	Signal signal;
	MagFields Bs{};

	// Take one forward Euler step if starting from the ground state or time step changed
	if (doForward)
	{
		std::cout << "No even time step file found. Doing one forward step." << std::endl;

		signal = getSignal(t);
		Bs.Bq = BqScale * signal.Bq;
		Bs.Bz = BzScale * signal.Bz;
		Bs.BqQuad = BqQuadScale * signal.Bq;
		Bs.BzQuad = BzQuadScale * signal.Bz;
		update_q << <dimGrid, edgeDimBlock >> > (d_oddQ, d_oddQ, d_oddPsi, d_d0, dimensions);
		forwardEuler << <dimGrid, psiDimBlock >> > (d_evenPsi, d_oddPsi, d_oddQ, d_d1, d_hodges, Bs, dimensions, block_scale, d_p0, c0, c2);
	}
	else
	{
		std::cout << "Skipping the forward step." << std::endl;
	}

	const double volume = block_scale * block_scale * block_scale * VOLUME;

#if COMPUTE_GROUND_STATE
	uint iter = 0;

	normalize_h(dimGrid, dimBlock, d_density, d_evenPsi, dimensions, bodies, volume);

	while (true)
	{
		if ((iter % 6000) == 0) std::cout << "Iteration " << iter << std::endl;
#if SAVE_PICTURE
		if ((iter % SAVE_FREQUENCY) == 0)
		{
			checkCudaErrors(cudaMemcpy3D(&evenPsiBackParams));
			drawDensity("GS", h_evenPsi, dxsize, dysize, dzsize, iter);
			printDensity(dimGrid, dimBlock, d_density, d_evenPsi, dimensions, bodies, volume);
		}
#endif
		if (iter == 100000)
		{
			polarState << <dimGrid, dimBlock >> > (d_evenPsi, dimensions);
			checkCudaErrors(cudaMemcpy3D(&evenPsiBackParams));
			std::ofstream fs(GROUND_STATE_FILENAME, std::ios::binary | std::ios_base::trunc);
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
	std::string times = std::string("times = [") + (loadGroundState ? "" : "times");
	std::string bqString = std::string("Bq = [") + (loadGroundState ? "" : "Bq");
	std::string bzString = std::string("Bz = [") + (loadGroundState ? "" : "Bz");
	std::string magX = std::string("mag_x = [") + (loadGroundState ? "" : "mag_x");
	std::string magY = std::string("mag_y = [") + (loadGroundState ? "" : "mag_y");
	std::string magZ = std::string("mag_z = [") + (loadGroundState ? "" : "mag_z");
	std::string densityStr = std::string("norm = [") + (loadGroundState ? "" : "norm");

	while (true)
	{
#if SAVE_PICTURE
		// Copy back from device memory to host memory
		checkCudaErrors(cudaMemcpy3D(&oddPsiBackParams));

		// Measure wall clock time
		static auto prevTime = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::high_resolution_clock::now() - prevTime;
		std::cout << "Simulation time: " << t << " ms. Real time from previous save: " << duration.count() * 1e-9 << " s." << std::endl;
		prevTime = std::chrono::high_resolution_clock::now();

#if RELATIVISTIC
		drawDensity("hyperbolic_", h_oddPsi, dxsize, dysize, dzsize, t);
#else
		drawDensity("parabolic_", h_oddPsi, dxsize, dysize, dzsize, t);
#endif

		//uvTheta << <dimGrid, psiDimBlock >> > (d_u, d_v, d_theta, d_oddPsi, dimensions);
		//cudaMemcpy(h_u, d_u, bodies * sizeof(double3), cudaMemcpyDeviceToHost);
		//cudaMemcpy(h_v, d_v, bodies * sizeof(double3), cudaMemcpyDeviceToHost);
		//cudaMemcpy(h_theta, d_theta, bodies * sizeof(double), cudaMemcpyDeviceToHost);
		//drawUtheta(h_u, h_theta, xsize, ysize, zsize, t);
#endif

		// integrate one iteration
		for (uint step = 0; step < SAVE_FREQUENCY; step++)
		{
#if RELATIVISTIC
			// update odd values (imaginary terms)
			t += dt / omega_r * 1e3; // [ms]
			signal = getSignal(t);
			Bs.Bq = BqScale * signal.Bq;
			Bs.Bz = BzScale * signal.Bz;
			Bs.BqQuad = BqQuadScale * signal.Bq;
			Bs.BzQuad = BzQuadScale * signal.Bz;
			update_psi << <dimGrid, psiDimBlock >> > (d_oddPsi, d_evenPsi, d_evenQ, d_d1, d_hodges, Bs, dimensions, block_scale, d_p0, c0, c2);
			update_q << <dimGrid, edgeDimBlock >> > (d_oddQ, d_evenQ, d_evenPsi, d_d0, dimensions);

			// update even values (real terms)
			t += dt / omega_r * 1e3; // [ms]
			signal = getSignal(t);
			Bs.Bq = BqScale * signal.Bq;
			Bs.Bz = BzScale * signal.Bz;
			Bs.BqQuad = BqQuadScale * signal.Bq;
			Bs.BzQuad = BzQuadScale * signal.Bz;
			update_psi << <dimGrid, psiDimBlock >> > (d_evenPsi, d_oddPsi, d_oddQ, d_d1, d_hodges, Bs, dimensions, block_scale, d_p0, c0, c2);
			update_q << <dimGrid, edgeDimBlock >> > (d_evenQ, d_oddQ, d_oddPsi, d_d0, dimensions);
#else
			// update odd values (imaginary terms)
			t += dt / omega_r * 1e3; // [ms]
			signal = getSignal(t);
			Bs.Bq = BqScale * signal.Bq;
			Bs.Bz = BzScale * signal.Bz;
			Bs.BqQuad = BqQuadScale * signal.Bq;
			Bs.BzQuad = BzQuadScale * signal.Bz;
			update_q << <dimGrid, edgeDimBlock >> > (d_evenQ, d_evenQ, d_evenPsi, d_d0, dimensions);
			update_psi << <dimGrid, psiDimBlock >> > (d_oddPsi, d_evenPsi, d_evenQ, d_d1, d_hodges, Bs, dimensions, block_scale, d_p0, c0, c2);

			// update even values (real terms)
			t += dt / omega_r * 1e3; // [ms]
			signal = getSignal(t);
			Bs.Bq = BqScale * signal.Bq;
			Bs.Bz = BzScale * signal.Bz;
			Bs.BqQuad = BqQuadScale * signal.Bq;
			Bs.BzQuad = BzQuadScale * signal.Bz;
			update_q << <dimGrid, edgeDimBlock >> > (d_oddQ, d_oddQ, d_oddPsi, d_d0, dimensions);
			update_psi << <dimGrid, psiDimBlock >> > (d_evenPsi, d_oddPsi, d_oddQ, d_d1, d_hodges, Bs, dimensions, block_scale, d_p0, c0, c2);
#endif
		}

#if SAVE_STATES
		// Copy back from device memory to host memory
		checkCudaErrors(cudaMemcpy3D(&oddPsiBackParams));

		// Measure wall clock time
		static auto prevTime = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::high_resolution_clock::now() - prevTime;
		std::cout << "Simulation time: " << t << " ms. Real time from previous save: " << duration.count() * 1e-9 << " s." << std::endl;

		drawDensity("", h_oddPsi, dxsize, dysize, dzsize, t - 200);

		cudaMemcpy(h_u, d_u, bodies * sizeof(double3), cudaMemcpyDeviceToHost);
		cudaMemcpy(h_v, d_v, bodies * sizeof(double3), cudaMemcpyDeviceToHost);
		cudaMemcpy(h_theta, d_theta, bodies * sizeof(double), cudaMemcpyDeviceToHost);

		double4 magDens = getMagnetizationAndDensity(dimGrid, dimBlock, d_magnetization, d_density, d_oddPsi, dimensions, bodies, volume);
		times += ", " + toString(t);
		bqString += ", " + toString(Bs.Bq);
		bzString += ", " + toString(Bs.Bz);
		magX += ", " + toString(magDens.x);
		magY += ", " + toString(magDens.y);
		magZ += ", " + toString(magDens.z);
		densityStr += ", " + toString(magDens.w);

		if ((int(t) % 5) == 0)
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
			textFile.save(toString(t) + ".m");

			std::ofstream oddFs(toString(t) + ".dat", std::ios::binary | std::ios_base::trunc);
			if (oddFs.fail() != 0) return 1;
			oddFs.write((char*)&h_oddPsi[0], hostSize * sizeof(BlockPsis));
			oddFs.close();

			checkCudaErrors(cudaMemcpy3D(&evenPsiBackParams));
			std::ofstream evenFs("even_" + toString(t) + ".dat", std::ios::binary | std::ios_base::trunc);
			if (evenFs.fail() != 0) return 1;
			evenFs.write((char*)&h_evenPsi[0], hostSize * sizeof(BlockPsis));
			evenFs.close();

			std::cout << "Saved the state!" << std::endl;

			times = std::string("times = [") + (loadGroundState ? "" : "times");
			bqString = std::string("Bq = [") + (loadGroundState ? "" : "Bq");
			bzString = std::string("Bz = [") + (loadGroundState ? "" : "Bz");
			magX = std::string("mag_x = [") + (loadGroundState ? "" : "mag_x");
			magY = std::string("mag_y = [") + (loadGroundState ? "" : "mag_y");
			magZ = std::string("mag_z = [") + (loadGroundState ? "" : "mag_z");
			densityStr = std::string("norm = [") + (loadGroundState ? "" : "norm");

			if (t > END_TIME)
			{
				return 0;
			}
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

int main(int argc, char** argv)
{
	const double blockScale = DOMAIN_SIZE_X / REPLICABLE_STRUCTURE_COUNT_X;

	std::cout << "Start simulating from t = " << t << " ms, with a time step size of " << dt << "." << std::endl;
	std::cout << "The simulation will end at " << END_TIME << " ms." << std::endl;
	std::cout << "Block scale = " << blockScale << std::endl;
	std::cout << "Dual edge length = " << DUAL_EDGE_LENGTH * blockScale << std::endl;

	// integrate in time using DEC
	auto domainMin = Vector3(-DOMAIN_SIZE_X * 0.5, -DOMAIN_SIZE_Y * 0.5, -DOMAIN_SIZE_Z * 0.5);
	auto domainMax = Vector3(DOMAIN_SIZE_X * 0.5, DOMAIN_SIZE_Y * 0.5, DOMAIN_SIZE_Z * 0.5);
	integrateInTime(blockScale, domainMin, domainMax);

	return 0;
}
