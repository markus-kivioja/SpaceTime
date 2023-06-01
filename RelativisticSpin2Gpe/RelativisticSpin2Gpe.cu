#include <cuda_runtime.h>
#include "helper_cuda.h"

#define Z_QUANTIZED 0
#define Y_QUANTIZED 1
#define X_QUANTIZED 2

#define BASIS Z_QUANTIZED

enum class Phase {
	UN = 0,
	BN_VERT,
	BN_HORI,
	CYCLIC
};
constexpr Phase initPhase = Phase::CYCLIC;

std::string phaseToString(Phase phase)
{
	switch (phase)
	{
	case Phase::UN:
		return "un";
	case Phase::BN_VERT:
		return "bn_vert";
	case Phase::BN_HORI:
		return "bn_hori";
	case Phase::CYCLIC:
		return "cyclic";
	default:
		return "";
	}
}

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

//#include "AliceRingRamps.h"
#include "KnotRamps.h"

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

#define HYPERBOLIC 1
#define PARABOLIC 0
#define COMPUTE_ERROR (HYPERBOLIC && PARABOLIC)

#define SAVE_STATES 0
#define SAVE_PICTURE 0

#define THREAD_BLOCK_X 16
#define THREAD_BLOCK_Y 2
#define THREAD_BLOCK_Z 1

constexpr double DOMAIN_SIZE_X = 20.0;
constexpr double DOMAIN_SIZE_Y = 20.0;
constexpr double DOMAIN_SIZE_Z = 20.0;

constexpr double REPLICABLE_STRUCTURE_COUNT_X = 58.0 + 8 * 6.0;
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
constexpr double a_0 = 87.9;
constexpr double a_2 = 91.41;
constexpr double a_4 = 98.36;

constexpr double atomMass = 1.44316060e-25;
constexpr double hbar = 1.05457148e-34; // [m^2 kg / s]
const double a_r = sqrt(hbar / (atomMass * omega_r)); //[m]

const double c0 = 4 * PI * N * (4 * a_2 + 3 * a_4) * a_bohr / (7 * a_r);
const double c2 = 4 * PI * N * (a_4 - a_2) * a_bohr / (7 * a_r);
const double c4 = 4 * PI * N * (7 * a_0 - 10 * a_2 + 3 * a_4) * a_bohr / (7 * a_r);

constexpr double muB = 9.27400968e-24; // [m^2 kg / s^2 T^-1] Bohr magneton

const double BqScale = (0.5 * muB / (hbar * omega_r) * a_r) / 100.; // [cm/Gauss]
constexpr double BzScale = (0.5 * muB / (hbar * omega_r)) / 10000.; // [1/Gauss]

constexpr double A_hfs = 3.41734130545215;
const double BqQuadScale = 100 * a_r * sqrt(0.25 * 1000 * (1.399624624 * 1.399624624) / (trapFreq_r * 2 * A_hfs)); //[cm/Gauss]
const double BzQuadScale = sqrt(0.25 * 1000 * (1.399624624 * 1.399624624) / (trapFreq_r * 2 * A_hfs)); //[1/Gauss]  \sqrt{g_q}

constexpr double SQRT_2 = 1.41421356237309;
//constexpr double INV_SQRT_2 = 0.70710678118655;

constexpr double NOISE_AMPLITUDE = 0;

double dt = 47e-4;
double dt_increse = 1e-5;

const double IMAGE_SAVE_INTERVAL = 0.05; // ms
uint IMAGE_SAVE_FREQUENCY = uint(IMAGE_SAVE_INTERVAL * 0.5 / 1e3 * omega_r / dt) + 1;

const uint STATE_SAVE_INTERVAL = 10.0; // ms

double t = 0; // Start time in ms
constexpr double END_TIME = 0.55; // End time in ms

#if COMPUTE_GROUND_STATE
double sigma = 0.1;
#else
double sigma = 1.0;
#endif
double dt_per_sigma = dt / sigma;

std::string toStringShort(const double value)
{
	std::ostringstream out;
	out.precision(2);
	out << std::fixed << value;
	return out.str();
};

const std::string EXTRA_INFORMATION = toStringShort(DOMAIN_SIZE_X) + "_" + toStringShort(REPLICABLE_STRUCTURE_COUNT_X) + "_" + phaseToString(initPhase);
const std::string GROUND_STATE_PSI_FILENAME = "ground_state_psi_" + EXTRA_INFORMATION + ".dat";
const std::string GROUND_STATE_Q_FILENAME = "ground_state_q_" + EXTRA_INFORMATION + ".dat";

__device__ __inline__ double trap(double3 p)
{
	double x = p.x * lambda_x;
	double y = p.y * lambda_y;
	double z = p.z * lambda_z;
	return 0.5 * (x * x + y * y + z * z) + 100.0;
}

__device__ __inline__ double3 magneticField(double3 p, double Bq, double3 Bb)
{
	return { Bq * p.x + Bb.x, Bq * p.y + Bb.y, -2 * Bq * p.z + Bb.z };
}

#include "utils.h"

__global__ void density(double* density, PitchedPtr prevStep, uint3 dimensions)
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
	Complex5Vec psi = ((BlockPsis*)pPsi)->values[dualNodeId];

	size_t idx = VALUES_IN_BLOCK * (zid * dimensions.x * dimensions.y + yid * dimensions.x + dataXid) + dualNodeId;
	density[idx] = (psi.s2 * conj(psi.s2)).x + (psi.s1 * conj(psi.s1)).x + (psi.s0 * conj(psi.s0)).x + (psi.s_1 * conj(psi.s_1)).x + (psi.s_2 * conj(psi.s_2)).x;
}

__global__ void weightedDiff(double* result, PitchedPtr pLeft, PitchedPtr pRight, uint3 dimensions)
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

	Complex5Vec left = ((BlockPsis*)(pLeft.ptr + pLeft.slicePitch * zid + pLeft.pitch * yid) + dataXid)->values[dualNodeId];
	Complex5Vec right = ((BlockPsis*)(pRight.ptr + pRight.slicePitch * zid + pRight.pitch * yid) + dataXid)->values[dualNodeId];

	Complex5Vec diff = { right.s2 - left.s2, right.s1 - left.s1, right.s0 - left.s0, right.s_1 - left.s_1, right.s_2 - left.s_2 };

	double leftSqr = (conj(left.s2) * left.s2).x + (conj(left.s1) * left.s1).x + (conj(left.s0) * left.s0).x + (conj(left.s_1) * left.s_1).x + (conj(left.s_2) * left.s_2).x;
	double diffSqr = (conj(diff.s2) * diff.s2).x + (conj(diff.s1) * diff.s1).x + (conj(diff.s0) * diff.s0).x + (conj(diff.s_1) * diff.s_1).x + (conj(diff.s_2) * diff.s_2).x;

	size_t idx = VALUES_IN_BLOCK * (zid * dimensions.x * dimensions.y + yid * dimensions.x + dataXid) + dualNodeId;
	result[idx] = leftSqr * diffSqr;
}

__global__ void integrate(double* dataVec, size_t stride, bool addLast, double dv)
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

	if (stride == 1)
	{
		dataVec[0] *= dv;
	}
}

__global__ void integrateVec(double3* dataVec, size_t stride, bool addLast, double dv)
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

	if (stride == 1)
	{
		dataVec[0] = dv * dataVec[0];
	}
}

__global__ void integrateVecWithDensity(double3* dataVec, double* density, size_t stride, bool addLast, double dv)
{
	size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx >= stride)
	{
		return;
	}

	dataVec[idx] = density[idx] * dataVec[idx] + density[idx + stride] * dataVec[idx + stride];

	if ((idx == (stride - 1)) && addLast)
	{
		dataVec[idx] += dataVec[idx + stride + 1];
	}

	if (stride == 1)
	{
		dataVec[0] = dv * dataVec[0];
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
	Complex5Vec psi = blockPsis->values[dualNodeId];
	double sqrtDens = sqrt(density[0]);
	psi.s2 = psi.s2 / sqrtDens;
	psi.s1 = psi.s1 / sqrtDens;
	psi.s0 = psi.s0 / sqrtDens;
	psi.s_1 = psi.s_1 / sqrtDens;
	psi.s_2 = psi.s_2 / sqrtDens;

	blockPsis->values[dualNodeId] = psi;
}

__global__ void unState(PitchedPtr psi, uint3 dimensions) // Uniaxial nematic phase
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

	Complex5Vec prev = pPsi->values[dualNodeId];

	double normSq_s2 = prev.s2.x * prev.s2.x + prev.s2.y * prev.s2.y;
	double normSq_s1 = prev.s1.x * prev.s1.x + prev.s1.y * prev.s1.y;
	double normSq_s0 = prev.s0.x * prev.s0.x + prev.s0.y * prev.s0.y;
	double normSq_s_1 = prev.s_1.x * prev.s_1.x + prev.s_1.y * prev.s_1.y;
	double normSq_s_2 = prev.s_2.x * prev.s_2.x + prev.s_2.y * prev.s_2.y;
	double normSq = normSq_s2 + normSq_s1 + normSq_s0 + normSq_s_1 + normSq_s_2;

	pPsi->values[dualNodeId].s2 = { 0, 0 };
	pPsi->values[dualNodeId].s1 = { 0, 0 };
	pPsi->values[dualNodeId].s0 = { sqrt(normSq), 0 };
	pPsi->values[dualNodeId].s_1 = { 0, 0 };
	pPsi->values[dualNodeId].s_2 = { 0, 0 };
};

__global__ void horizontalBnState(PitchedPtr psi, uint3 dimensions, double phase = 0) // Horizontal orientation of the biaxial nematic phase
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

	Complex5Vec prev = pPsi->values[dualNodeId];

	double normSq_s2 = prev.s2.x * prev.s2.x + prev.s2.y * prev.s2.y;
	double normSq_s1 = prev.s1.x * prev.s1.x + prev.s1.y * prev.s1.y;
	double normSq_s0 = prev.s0.x * prev.s0.x + prev.s0.y * prev.s0.y;
	double normSq_s_1 = prev.s_1.x * prev.s_1.x + prev.s_1.y * prev.s_1.y;
	double normSq_s_2 = prev.s_2.x * prev.s_2.x + prev.s_2.y * prev.s_2.y;
	double normSq = normSq_s2 + normSq_s1 + normSq_s0 + normSq_s_1 + normSq_s_2;

	double amplitude = sqrt(normSq / 2);

	pPsi->values[dualNodeId].s2 = { amplitude, 0 };
	pPsi->values[dualNodeId].s1 = { 0, 0 };
	pPsi->values[dualNodeId].s0 = { 0, 0 };
	pPsi->values[dualNodeId].s_1 = { 0, 0 };
	pPsi->values[dualNodeId].s_2 = { cos(phase) * amplitude, sin(phase) * amplitude };
};

__global__ void verticalBnState(PitchedPtr psi, uint3 dimensions, double phase = 0) // Vertical orientation of the biaxial nematic phase
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

	Complex5Vec prev = pPsi->values[dualNodeId];

	double normSq_s2 = prev.s2.x * prev.s2.x + prev.s2.y * prev.s2.y;
	double normSq_s1 = prev.s1.x * prev.s1.x + prev.s1.y * prev.s1.y;
	double normSq_s0 = prev.s0.x * prev.s0.x + prev.s0.y * prev.s0.y;
	double normSq_s_1 = prev.s_1.x * prev.s_1.x + prev.s_1.y * prev.s_1.y;
	double normSq_s_2 = prev.s_2.x * prev.s_2.x + prev.s_2.y * prev.s_2.y;
	double normSq = normSq_s2 + normSq_s1 + normSq_s0 + normSq_s_1 + normSq_s_2;

	double amplitude = sqrt(normSq / 2);

	pPsi->values[dualNodeId].s2 = { 0, 0 };
	pPsi->values[dualNodeId].s1 = { amplitude, 0 };
	pPsi->values[dualNodeId].s0 = { 0, 0 };
	pPsi->values[dualNodeId].s_1 = { cos(phase) * amplitude, sin(phase) * amplitude };
	pPsi->values[dualNodeId].s_2 = { 0, 0 };
};

__global__ void cyclicState(PitchedPtr psi, uint3 dimensions, double phase = 0) // Cyclic phase
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

	Complex5Vec prev = pPsi->values[dualNodeId];

	double normSq_s2 = prev.s2.x * prev.s2.x + prev.s2.y * prev.s2.y;
	double normSq_s1 = prev.s1.x * prev.s1.x + prev.s1.y * prev.s1.y;
	double normSq_s0 = prev.s0.x * prev.s0.x + prev.s0.y * prev.s0.y;
	double normSq_s_1 = prev.s_1.x * prev.s_1.x + prev.s_1.y * prev.s_1.y;
	double normSq_s_2 = prev.s_2.x * prev.s_2.x + prev.s_2.y * prev.s_2.y;
	double normSq = normSq_s2 + normSq_s1 + normSq_s0 + normSq_s_1 + normSq_s_2;

	double amplitude_m2 = sqrt(normSq * 1 / 3);
	double amplitude_m_1 = sqrt(normSq * 2 / 3);

	pPsi->values[dualNodeId].s2 = { amplitude_m2, 0 };
	pPsi->values[dualNodeId].s1 = { 0, 0 };
	pPsi->values[dualNodeId].s0 = { 0, 0 };
	pPsi->values[dualNodeId].s_1 = { cos(phase) * amplitude_m_1, sin(phase) * amplitude_m_1 };
	pPsi->values[dualNodeId].s_2 = { 0, 0 };
};

#if COMPUTE_GROUND_STATE
__global__ void itp_q(PitchedPtr next_q, PitchedPtr prev_q, PitchedPtr psi, int3* d0, uint3 dimensions, double dt_per_sigma)
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

	Complex5Vec thisPsi = ((BlockPsis*)(pPsi))->values[d0[dualEdgeId].x];
	Complex5Vec otherPsi = ((BlockPsis*)(pPsi + d0[dualEdgeId].y))->values[d0[dualEdgeId].z];
	Complex5Vec d0psi;
	d0psi.s2 = otherPsi.s2 - thisPsi.s2;
	d0psi.s1 = otherPsi.s1 - thisPsi.s1;
	d0psi.s0 = otherPsi.s0 - thisPsi.s0;
	d0psi.s_1 = otherPsi.s_1 - thisPsi.s_1;
	d0psi.s_2 = otherPsi.s_2 - thisPsi.s_2;

	BlockEdges* next = (BlockEdges*)(next_q.ptr + next_q.slicePitch * zid + next_q.pitch * yid) + dataXid;

	BlockEdges* prev = (BlockEdges*)(prev_q.ptr + prev_q.slicePitch * zid + prev_q.pitch * yid) + dataXid;

	Complex5Vec q;
	q.s2 = dt_per_sigma * (-d0psi.s2 + prev->values[dualEdgeId].s2);
	q.s1 = dt_per_sigma * (-d0psi.s1 + prev->values[dualEdgeId].s1);
	q.s0 = dt_per_sigma * (-d0psi.s0 + prev->values[dualEdgeId].s0);
	q.s_1 = dt_per_sigma * (-d0psi.s_1 + prev->values[dualEdgeId].s_1);
	q.s_2 = dt_per_sigma * (-d0psi.s_2 + prev->values[dualEdgeId].s_2);

	next->values[dualEdgeId].s2 = prev->values[dualEdgeId].s2 - q.s2;
	next->values[dualEdgeId].s1 = prev->values[dualEdgeId].s1 - q.s1;
	next->values[dualEdgeId].s0 = prev->values[dualEdgeId].s0 - q.s0;
	next->values[dualEdgeId].s_1 = prev->values[dualEdgeId].s_1 - q.s_1;
	next->values[dualEdgeId].s_2 = prev->values[dualEdgeId].s_2 - q.s_2;
}

__global__ void itp_psi(PitchedPtr nextStep, PitchedPtr prevStep, PitchedPtr qs, const int2* __restrict__ d1Ptr, const double* __restrict__ hodges, const uint3 dimensions, const double block_scale, const double3 p0, const double c0, const double c2, const double c4, const double dt)
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
	const Complex5Vec prev = ((BlockPsis*)prevPsi)->values[dualNodeId];

	Complex5Vec H;
	H.s2 = { 0, 0 };
	H.s1 = { 0, 0 };
	H.s0 = { 0, 0 };
	H.s_1 = { 0, 0 };
	H.s_2 = { 0, 0 };

	// Add the the second exterior derivative (d1 of d0) to the Hamiltonian
	uint startEdgeId = dualNodeId * FACE_COUNT;
#pragma unroll
	for (int edgeIdOffset = 0; edgeIdOffset < FACE_COUNT; ++edgeIdOffset)
	{
		int edgeId = startEdgeId + edgeIdOffset;
		int2 d1 = d1Ptr[edgeId];
		Complex5Vec d0psi = ((BlockEdges*)(qPtr + d1.x))->values[d1.y];
		const double hodge = hodges[edgeId];

		H.s2 += hodge * d0psi.s2;
		H.s1 += hodge * d0psi.s1;
		H.s0 += hodge * d0psi.s0;
		H.s_1 += hodge * d0psi.s_1;
		H.s_2 += hodge * d0psi.s_2;
	}

	const double normSq_s2 = prev.s2.x * prev.s2.x + prev.s2.y * prev.s2.y;
	const double normSq_s1 = prev.s1.x * prev.s1.x + prev.s1.y * prev.s1.y;
	const double normSq_s0 = prev.s0.x * prev.s0.x + prev.s0.y * prev.s0.y;
	const double normSq_s_1 = prev.s_1.x * prev.s_1.x + prev.s_1.y * prev.s_1.y;
	const double normSq_s_2 = prev.s_2.x * prev.s_2.x + prev.s_2.y * prev.s_2.y;
	const double normSq = normSq_s2 + normSq_s1 + normSq_s0 + normSq_s_1 + normSq_s_2;

	const double3 localPos = d_localPos[dualNodeId];
	const double3 globalPos = { p0.x + block_scale * (dataXid * BLOCK_WIDTH_X + localPos.x),
		p0.y + block_scale * (yid * BLOCK_WIDTH_Y + localPos.y),
		p0.z + block_scale * (zid * BLOCK_WIDTH_Z + localPos.z) };

	double2 ab = { trap(globalPos) + c0 * normSq, 0 };

	double3 B = { 0 };

	const double Fz = c2 * (2.0 * normSq_s2 + normSq_s1 - normSq_s_1 - 2.0 * normSq_s_2);

	Complex5Vec diagonalTerm;
	diagonalTerm.s2 = double2{ 2.0 * Fz + 0.4 * c4 * normSq_s_2 - 2.0 * B.z, 0 } + ab;
	diagonalTerm.s1 = double2{ Fz + 0.4 * c4 * normSq_s_1 - B.z, 0 } + ab;
	diagonalTerm.s0 = double2{ 0.2 * c4 * normSq_s0             , 0 } + ab;
	diagonalTerm.s_1 = double2{ -Fz + 0.4 * c4 * normSq_s1 + B.z, 0 } + ab;
	diagonalTerm.s_2 = double2{ -2.0 * Fz + 0.4 * c4 * normSq_s2 + 2.0 * B.z, 0 } + ab;

	H.s2 += diagonalTerm.s2 * prev.s2;    // psi1
	H.s1 += diagonalTerm.s1 * prev.s1;    // psi2
	H.s0 += diagonalTerm.s0 * prev.s0;    // psi3
	H.s_1 += diagonalTerm.s_1 * prev.s_1; // psi4
	H.s_2 += diagonalTerm.s_2 * prev.s_2; // psi5

	double2 denominator = c2 * (2.0 * (prev.s2 * conj(prev.s1) +
		prev.s_1 * conj(prev.s_2)) +
		sqrt(6.0) * (prev.s1 * conj(prev.s0) +
			prev.s0 * conj(prev.s_1))) - double2{ B.x, -B.y };

	double2 c12 = denominator - 0.4 * c4 * prev.s_1 * conj(prev.s_2);
	double2 c45 = denominator - 0.4 * c4 * prev.s2 * conj(prev.s1);
	double2 c13 = 0.2 * c4 * prev.s0 * conj(prev.s_2);
	double2 c35 = 0.2 * c4 * prev.s2 * conj(prev.s0);
	double2 c23 = sqrt(1.5) * denominator - 0.2 * c4 * prev.s0 * conj(prev.s_1);
	double2 c34 = sqrt(1.5) * denominator - 0.2 * c4 * prev.s1 * conj(prev.s0);

	H.s2 += (c12 * prev.s1 + c13 * prev.s0);
	H.s1 += (conj(c12) * prev.s2 + c23 * prev.s0);
	H.s0 += (conj(c13) * prev.s2 + c35 * prev.s_2 + c34 * prev.s_1 + conj(c23) * prev.s1);
	H.s_1 += (conj(c34) * prev.s0 + c45 * prev.s_2);
	H.s_2 += (conj(c35) * prev.s0 + conj(c45) * prev.s_1);

	nextPsi->values[dualNodeId].s2 = prev.s2 - dt * H.s2;
	nextPsi->values[dualNodeId].s1 = prev.s1 - dt * H.s1;
	nextPsi->values[dualNodeId].s0 = prev.s0 - dt * H.s0;
	nextPsi->values[dualNodeId].s_1 = prev.s_1 - dt * H.s_1;
	nextPsi->values[dualNodeId].s_2 = prev.s_2 - dt * H.s_2;
};
#else
__global__ void forwardEuler_q(PitchedPtr next_q, PitchedPtr prev_q, PitchedPtr psi, int3* d0, uint3 dimensions, double dt_per_sigma, bool hyperb)
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

	Complex5Vec thisPsi = ((BlockPsis*)(pPsi))->values[d0[dualEdgeId].x];
	Complex5Vec otherPsi = ((BlockPsis*)(pPsi + d0[dualEdgeId].y))->values[d0[dualEdgeId].z];
	Complex5Vec d0psi;
	d0psi.s2 = otherPsi.s2 - thisPsi.s2;
	d0psi.s1 = otherPsi.s1 - thisPsi.s1;
	d0psi.s0 = otherPsi.s0 - thisPsi.s0;
	d0psi.s_1 = otherPsi.s_1 - thisPsi.s_1;
	d0psi.s_2 = otherPsi.s_2 - thisPsi.s_2;

	BlockEdges* next = (BlockEdges*)(next_q.ptr + next_q.slicePitch * zid + next_q.pitch * yid) + dataXid;

	if (hyperb)
	{
		BlockEdges* prev = (BlockEdges*)(prev_q.ptr + prev_q.slicePitch * zid + prev_q.pitch * yid) + dataXid;

		Complex5Vec q;
		q.s2 = dt_per_sigma * (d0psi.s2 - prev->values[dualEdgeId].s2);
		q.s1 = dt_per_sigma * (d0psi.s1 - prev->values[dualEdgeId].s1);
		q.s0 = dt_per_sigma * (d0psi.s0 - prev->values[dualEdgeId].s0);
		q.s_1 = dt_per_sigma * (d0psi.s_1 - prev->values[dualEdgeId].s_1);
		q.s_2 = dt_per_sigma * (d0psi.s_2 - prev->values[dualEdgeId].s_2);

		next->values[dualEdgeId].s2 = prev->values[dualEdgeId].s2 + make_double2(q.s2.y, -q.s2.x);
		next->values[dualEdgeId].s1 = prev->values[dualEdgeId].s1 + make_double2(q.s1.y, -q.s1.x);
		next->values[dualEdgeId].s0 = prev->values[dualEdgeId].s0 + make_double2(q.s0.y, -q.s0.x);
		next->values[dualEdgeId].s_1 = prev->values[dualEdgeId].s_1 + make_double2(q.s_1.y, -q.s_1.x);
		next->values[dualEdgeId].s_2 = prev->values[dualEdgeId].s_2 + make_double2(q.s_2.y, -q.s_2.x);
	}
	else
	{
		next->values[dualEdgeId] = d0psi;
	}
}

__global__ void forwardEuler(PitchedPtr nextStep, PitchedPtr prevStep, PitchedPtr qs, const int2* __restrict__ d1Ptr, const double* __restrict__ hodges, MagFields Bs, const uint3 dimensions, const double block_scale, const double3 p0, const double c0, const double c2, const double c4, double dt, bool hyperb)
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
	const Complex5Vec prev = ((BlockPsis*)prevPsi)->values[dualNodeId];

	Complex5Vec H;
	H.s2 = { 0, 0 };
	H.s1 = { 0, 0 };
	H.s0 = { 0, 0 };
	H.s_1 = { 0, 0 };
	H.s_2 = { 0, 0 };

	// Add the the second exterior derivative (d1 of d0) to the Hamiltonian
	uint startEdgeId = dualNodeId * FACE_COUNT;
#pragma unroll
	for (int edgeIdOffset = 0; edgeIdOffset < FACE_COUNT; ++edgeIdOffset)
	{
		int edgeId = startEdgeId + edgeIdOffset;
		int2 d1 = d1Ptr[edgeId];
		Complex5Vec d0psi = ((BlockEdges*)(qPtr + d1.x))->values[d1.y];
		const double hodge = hodges[edgeId];

		H.s2 += hodge * d0psi.s2;
		H.s1 += hodge * d0psi.s1;
		H.s0 += hodge * d0psi.s0;
		H.s_1 += hodge * d0psi.s_1;
		H.s_2 += hodge * d0psi.s_2;
	}
	if (hyperb)
	{
		H.s2 = -H.s2;
		H.s1 = -H.s1;
		H.s0 = -H.s0;
		H.s_1 = -H.s_1;
		H.s_2 = -H.s_2;
	}

	const double normSq_s2 = prev.s2.x * prev.s2.x + prev.s2.y * prev.s2.y;
	const double normSq_s1 = prev.s1.x * prev.s1.x + prev.s1.y * prev.s1.y;
	const double normSq_s0 = prev.s0.x * prev.s0.x + prev.s0.y * prev.s0.y;
	const double normSq_s_1 = prev.s_1.x * prev.s_1.x + prev.s_1.y * prev.s_1.y;
	const double normSq_s_2 = prev.s_2.x * prev.s_2.x + prev.s_2.y * prev.s_2.y;
	const double normSq = normSq_s2 + normSq_s1 + normSq_s0 + normSq_s_1 + normSq_s_2;

	const double3 localPos = d_localPos[dualNodeId];
	const double3 globalPos = { p0.x + block_scale * (dataXid * BLOCK_WIDTH_X + localPos.x),
		p0.y + block_scale * (yid * BLOCK_WIDTH_Y + localPos.y),
		p0.z + block_scale * (zid * BLOCK_WIDTH_Z + localPos.z) };

	double2 ab = { trap(globalPos) + c0 * normSq, 0 };

	double3 B = magneticField(globalPos, Bs.Bq, Bs.Bb);

	const double Fz = c2 * (2.0 * normSq_s2 + normSq_s1 - normSq_s_1 - 2.0 * normSq_s_2);

	Complex5Vec diagonalTerm;
	diagonalTerm.s2 = double2{ 2.0 * Fz + 0.4 * c4 * normSq_s_2 - 2.0 * B.z, 0 } + ab;
	diagonalTerm.s1 = double2{ Fz + 0.4 * c4 * normSq_s_1 - B.z, 0 } + ab;
	diagonalTerm.s0 = double2{ 0.2 * c4 * normSq_s0             , 0 } + ab;
	diagonalTerm.s_1 = double2{ -Fz + 0.4 * c4 * normSq_s1 + B.z, 0 } + ab;
	diagonalTerm.s_2 = double2{ -2.0 * Fz + 0.4 * c4 * normSq_s2 + 2.0 * B.z, 0 } + ab;

	H.s2 += diagonalTerm.s2 * prev.s2;    // psi1
	H.s1 += diagonalTerm.s1 * prev.s1;    // psi2
	H.s0 += diagonalTerm.s0 * prev.s0;    // psi3
	H.s_1 += diagonalTerm.s_1 * prev.s_1; // psi4
	H.s_2 += diagonalTerm.s_2 * prev.s_2; // psi5

	double2 denominator = c2 * (2.0 * (prev.s2 * conj(prev.s1) +
		prev.s_1 * conj(prev.s_2)) +
		sqrt(6.0) * (prev.s1 * conj(prev.s0) +
			prev.s0 * conj(prev.s_1))) - double2{ B.x, -B.y };

	double2 c12 = denominator - 0.4 * c4 * prev.s_1 * conj(prev.s_2);
	double2 c45 = denominator - 0.4 * c4 * prev.s2 * conj(prev.s1);
	double2 c13 = 0.2 * c4 * prev.s0 * conj(prev.s_2);
	double2 c35 = 0.2 * c4 * prev.s2 * conj(prev.s0);
	double2 c23 = sqrt(1.5) * denominator - 0.2 * c4 * prev.s0 * conj(prev.s_1);
	double2 c34 = sqrt(1.5) * denominator - 0.2 * c4 * prev.s1 * conj(prev.s0);

	H.s2 += (c12 * prev.s1 + c13 * prev.s0);
	H.s1 += (conj(c12) * prev.s2 + c23 * prev.s0);
	H.s0 += (conj(c13) * prev.s2 + c35 * prev.s_2 + c34 * prev.s_1 + conj(c23) * prev.s1);
	H.s_1 += (conj(c34) * prev.s0 + c45 * prev.s_2);
	H.s_2 += (conj(c35) * prev.s0 + conj(c45) * prev.s_1);

	nextPsi->values[dualNodeId].s2 = prev.s2 + dt * double2{ H.s2.y, -H.s2.x };
	nextPsi->values[dualNodeId].s1 = prev.s1 + dt * double2{ H.s1.y, -H.s1.x };
	nextPsi->values[dualNodeId].s0 = prev.s0 + dt * double2{ H.s0.y, -H.s0.x };
	nextPsi->values[dualNodeId].s_1 = prev.s_1 + dt * double2{ H.s_1.y, -H.s_1.x };
	nextPsi->values[dualNodeId].s_2 = prev.s_2 + dt * double2{ H.s_2.y, -H.s_2.x };
};

__global__ void update_q(PitchedPtr next_q, PitchedPtr prev_q, PitchedPtr psi, int3* d0, uint3 dimensions, double dt_per_sigma, bool hyperb)
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

	Complex5Vec thisPsi = ((BlockPsis*)(pPsi))->values[d0[dualEdgeId].x];
	Complex5Vec otherPsi = ((BlockPsis*)(pPsi + d0[dualEdgeId].y))->values[d0[dualEdgeId].z];
	Complex5Vec d0psi;
	d0psi.s2 = otherPsi.s2 - thisPsi.s2;
	d0psi.s1 = otherPsi.s1 - thisPsi.s1;
	d0psi.s0 = otherPsi.s0 - thisPsi.s0;
	d0psi.s_1 = otherPsi.s_1 - thisPsi.s_1;
	d0psi.s_2 = otherPsi.s_2 - thisPsi.s_2;

	BlockEdges* next = (BlockEdges*)(next_q.ptr + next_q.slicePitch * zid + next_q.pitch * yid) + dataXid;

	if (hyperb)
	{
		BlockEdges* prev = (BlockEdges*)(prev_q.ptr + prev_q.slicePitch * zid + prev_q.pitch * yid) + dataXid;

		Complex5Vec q;
		q.s2 = 2 * dt_per_sigma * (d0psi.s2 - prev->values[dualEdgeId].s2);
		q.s1 = 2 * dt_per_sigma * (d0psi.s1 - prev->values[dualEdgeId].s1);
		q.s0 = 2 * dt_per_sigma * (d0psi.s0 - prev->values[dualEdgeId].s0);
		q.s_1 = 2 * dt_per_sigma * (d0psi.s_1 - prev->values[dualEdgeId].s_1);
		q.s_2 = 2 * dt_per_sigma * (d0psi.s_2 - prev->values[dualEdgeId].s_2);

		next->values[dualEdgeId].s2 += make_double2(q.s2.y, -q.s2.x);
		next->values[dualEdgeId].s1 += make_double2(q.s1.y, -q.s1.x);
		next->values[dualEdgeId].s0 += make_double2(q.s0.y, -q.s0.x);
		next->values[dualEdgeId].s_1 += make_double2(q.s_1.y, -q.s_1.x);
		next->values[dualEdgeId].s_2 += make_double2(q.s_2.y, -q.s_2.x);
	}
	else
	{
		next->values[dualEdgeId] = d0psi;
	}
}

__global__ void update_psi(PitchedPtr nextStep, PitchedPtr prevStep, PitchedPtr qs, const int2* __restrict__ d1Ptr, const double* __restrict__ hodges, MagFields Bs, const uint3 dimensions, const double block_scale, const double3 p0, const double c0, const double c2, const double c4, double dt, bool hyperb)
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
	const Complex5Vec prev = ((BlockPsis*)prevPsi)->values[dualNodeId];

	Complex5Vec H;
	H.s2 = { 0, 0 };
	H.s1 = { 0, 0 };
	H.s0 = { 0, 0 };
	H.s_1 = { 0, 0 };
	H.s_2 = { 0, 0 };

	// Add the the second exterior derivative (d1 of d0) to the Hamiltonian
	uint startEdgeId = dualNodeId * FACE_COUNT;
#pragma unroll
	for (int edgeIdOffset = 0; edgeIdOffset < FACE_COUNT; ++edgeIdOffset)
	{
		int edgeId = startEdgeId + edgeIdOffset;
		int2 d1 = d1Ptr[edgeId];
		Complex5Vec d0psi = ((BlockEdges*)(qPtr + d1.x))->values[d1.y];
		const double hodge = hodges[edgeId];

		H.s2  += hodge * d0psi.s2;
		H.s1  += hodge * d0psi.s1;
		H.s0  += hodge * d0psi.s0;
		H.s_1 += hodge * d0psi.s_1;
		H.s_2 += hodge * d0psi.s_2;
	}
	if (hyperb)
	{
		H.s2 = -H.s2;
		H.s1 = -H.s1;
		H.s0 = -H.s0;
		H.s_1 = -H.s_1;
		H.s_2 = -H.s_2;
	}

	const double normSq_s2 = prev.s2.x * prev.s2.x + prev.s2.y * prev.s2.y;
	const double normSq_s1 = prev.s1.x * prev.s1.x + prev.s1.y * prev.s1.y;
	const double normSq_s0 = prev.s0.x * prev.s0.x + prev.s0.y * prev.s0.y;
	const double normSq_s_1 = prev.s_1.x * prev.s_1.x + prev.s_1.y * prev.s_1.y;
	const double normSq_s_2 = prev.s_2.x * prev.s_2.x + prev.s_2.y * prev.s_2.y;
	const double normSq = normSq_s2 + normSq_s1 + normSq_s0 + normSq_s_1 + normSq_s_2;

	const double3 localPos = d_localPos[dualNodeId];
	const double3 globalPos = { p0.x + block_scale * (dataXid * BLOCK_WIDTH_X + localPos.x),
		p0.y + block_scale * (yid * BLOCK_WIDTH_Y + localPos.y),
		p0.z + block_scale * (zid * BLOCK_WIDTH_Z + localPos.z) };

	double2 ab = { trap(globalPos) + c0 * normSq, 0 };

	double3 B = magneticField(globalPos, Bs.Bq, Bs.Bb);

	const double Fz = c2 * (2.0 * normSq_s2 + normSq_s1 - normSq_s_1 - 2.0 * normSq_s_2);

	Complex5Vec diagonalTerm;
	diagonalTerm.s2 = double2{  2.0 * Fz + 0.4 * c4 * normSq_s_2 - 2.0 * B.z, 0 } + ab;
	diagonalTerm.s1 = double2{        Fz + 0.4 * c4 * normSq_s_1 -       B.z, 0 } + ab;
	diagonalTerm.s0 = double2{             0.2 * c4 * normSq_s0             , 0 } + ab;
	diagonalTerm.s_1 = double2{      -Fz + 0.4 * c4 * normSq_s1  +       B.z, 0 } + ab;
	diagonalTerm.s_2 = double2{-2.0 * Fz + 0.4 * c4 * normSq_s2  + 2.0 * B.z, 0 } + ab;

	H.s2 += diagonalTerm.s2 * prev.s2;    // psi1
	H.s1 += diagonalTerm.s1 * prev.s1;    // psi2
	H.s0 += diagonalTerm.s0 * prev.s0;    // psi3
	H.s_1 += diagonalTerm.s_1 * prev.s_1; // psi4
	H.s_2 += diagonalTerm.s_2 * prev.s_2; // psi5

	double2 denominator = c2 * (2.0 * (prev.s2  * conj(prev.s1) +
		                               prev.s_1 * conj(prev.s_2)) +
		                  sqrt(6.0) * (prev.s1  * conj(prev.s0) +
			                           prev.s0  * conj(prev.s_1))) - double2{ B.x, -B.y };

	double2 c12 = denominator - 0.4 * c4 * prev.s_1 * conj(prev.s_2);
	double2 c45 = denominator - 0.4 * c4 * prev.s2 * conj(prev.s1);
	double2 c13 = 0.2 * c4 * prev.s0 * conj(prev.s_2);
	double2 c35 = 0.2 * c4 * prev.s2 * conj(prev.s0);
	double2 c23 = sqrt(1.5) * denominator - 0.2 * c4 * prev.s0 * conj(prev.s_1);
	double2 c34 = sqrt(1.5) * denominator - 0.2 * c4 * prev.s1 * conj(prev.s0);

	H.s2  += (c12 * prev.s1 + c13 * prev.s0);
	H.s1  += (conj(c12) * prev.s2 + c23 * prev.s0);
	H.s0  += (conj(c13) * prev.s2 + c35 * prev.s_2 + c34 * prev.s_1 + conj(c23) * prev.s1);
	H.s_1 += (conj(c34) * prev.s0 + c45 * prev.s_2);
	H.s_2 += (conj(c35) * prev.s0 + conj(c45) * prev.s_1);

	nextPsi->values[dualNodeId].s2 += 2 * dt * double2{ H.s2.y, -H.s2.x };
	nextPsi->values[dualNodeId].s1 += 2 * dt * double2{ H.s1.y, -H.s1.x };
	nextPsi->values[dualNodeId].s0 += 2 * dt * double2{ H.s0.y, -H.s0.x };
	nextPsi->values[dualNodeId].s_1 += 2 * dt * double2{ H.s_1.y, -H.s_1.x };
	nextPsi->values[dualNodeId].s_2 += 2 * dt * double2{ H.s_2.y, -H.s_2.x };
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

	std::cout << "Dual 0-cells in a replicable structure: " << bsize << std::endl;
	std::cout << "Replicable structure instances in x: " << xsize << ", y: " << ysize << ", z: " << zsize << std::endl;
	uint64_t bodies = xsize * ysize * zsize * bsize;
	std::cout << "Dual 0-cells in total: " << bodies << std::endl;

	// Initialize device memory
	const size_t dxsize = xsize + 2; // One element buffer to both ends
	const size_t dysize = ysize + 2; // One element buffer to both ends
	const size_t dzsize = zsize + 2; // One element buffer to both ends
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

	//double* d_energy = allocDevice<double>(bodies);
	double* d_density = allocDevice<double>(bodies);
#if COMPUTE_ERROR
	double* d_error = allocDevice<double>(bodies);
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

		std::default_random_engine generator(std::chrono::high_resolution_clock::now().time_since_epoch().count());
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
						const double2 s2{ distribution(generator), distribution(generator) };
						const double2 s1{ distribution(generator), distribution(generator) };
						const double2 s0{ distribution(generator), distribution(generator) };
						const double2 s_1{ distribution(generator), distribution(generator) };
						const double2 s_2{ distribution(generator), distribution(generator) };
						h_evenPsiHyper[dstI].values[l].s2 = s2;
						h_evenPsiHyper[dstI].values[l].s1 = s1;
						h_evenPsiHyper[dstI].values[l].s0 = s0;
						h_evenPsiHyper[dstI].values[l].s_1 = s_1;
						h_evenPsiHyper[dstI].values[l].s_2 = s_2;
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
	dim3 psiDimBlock(THREAD_BLOCK_X* VALUES_IN_BLOCK, THREAD_BLOCK_Y, THREAD_BLOCK_Z);
	dim3 edgeDimBlock(THREAD_BLOCK_X* EDGES_IN_BLOCK, THREAD_BLOCK_Y, THREAD_BLOCK_Z);
	dim3 dimGrid((xsize + THREAD_BLOCK_X - 1) / THREAD_BLOCK_X,
		(ysize + THREAD_BLOCK_Y - 1) / THREAD_BLOCK_Y,
		((zsize + THREAD_BLOCK_Z - 1) / THREAD_BLOCK_Z));

	Signal signal;
	MagFields Bs{ 0 };

	const double volume = block_scale * block_scale * block_scale * VOLUME;

#if COMPUTE_GROUND_STATE
	if (continueFromEarlier)
	{
		const double PHASE = 0;// PI / 2;

		switch (initPhase)
		{
		case Phase::UN:
			std::cout << "Transform ground state to uniaxial nematic phase." << std::endl;
			unState << <dimGrid, psiDimBlock >> > (d_evenPsiHyper, dimensions);
			break;
		case Phase::BN_VERT:
			std::cout << "Transform ground state to vertically oriented biaxial nematic phase with a phase of " << PHASE << "." << std::endl;
			verticalBnState << <dimGrid, psiDimBlock >> > (d_evenPsiHyper, dimensions, PHASE);
			break;
		case Phase::BN_HORI:
			std::cout << "Transform ground state to horizontally oriented biaxial nematic phase with a phase of " << PHASE << "." << std::endl;
			horizontalBnState << <dimGrid, psiDimBlock >> > (d_evenPsiHyper, dimensions, PHASE);
			break;
		case Phase::CYCLIC:
			std::cout << "Transform ground state to cyclic phase with a phase of " << PHASE << "." << std::endl;
			cyclicState << <dimGrid, psiDimBlock >> > (d_evenPsiHyper, dimensions, PHASE);
			break;
		default:
			std::cout << "Initial phase " << (int)initPhase << " is not supported!";
			break;
		}

		std::cout << getDensity(dimGrid, psiDimBlock, d_density, d_evenPsiHyper, dimensions, bodies, volume) << std::endl;
	}
#endif
	double extraPot = 0.0;

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
		forwardEuler << <dimGrid, psiDimBlock >> > (d_evenPsiHyper, d_oddPsiHyper, d_oddQHyper, d_d1, d_hodges, Bs, dimensions, block_scale, d_p0, c0, c2, c4, dt, true);
		forwardEuler_q << <dimGrid, edgeDimBlock >> > (d_evenQHyper, d_oddQHyper, d_oddPsiHyper, d_d0, dimensions, dt_per_sigma, true);
#endif
#if PARABOLIC
		forwardEuler_q << <dimGrid, edgeDimBlock >> > (d_oddQPara, d_oddQPara, d_oddPsiPara, d_d0, dimensions, dt_per_sigma, false);
		forwardEuler << <dimGrid, psiDimBlock >> > (d_evenPsiPara, d_oddPsiPara, d_oddQPara, d_d1, d_hodges, Bs, dimensions, block_scale, d_p0, c0, c2, c4, dt, false);
		forwardEuler_q << <dimGrid, edgeDimBlock >> > (d_evenQPara, d_evenQPara, d_evenPsiPara, d_d0, dimensions, dt_per_sigma, false);
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
		itp_q << <dimGrid, edgeDimBlock >> > (d_evenQHyper, d_evenQHyper, d_evenPsiHyper, d_d0, dimensions, dt_per_sigma);
		itp_q << <dimGrid, edgeDimBlock >> > (d_oddQHyper, d_oddQHyper, d_oddPsiHyper, d_d0, dimensions, dt_per_sigma);
	}

	while (true)
	{
		if ((iter % 1000) == 0) std::cout << "Iteration " << iter << std::endl;
#if SAVE_PICTURE
		if ((iter % 1000) == 0)
		{
			checkCudaErrors(cudaMemcpy3D(&evenPsiBackParamsHyper));
			signal = getSignal(0);
			Bs.Bq = BqScale * signal.Bq;
			Bs.Bb = BzScale * signal.Bb;
			drawIandR(folder, h_evenPsiHyper, dxsize, dysize, dzsize, iter, Bs, d_p0, block_scale);
			std::cout << getDensity(dimGrid, psiDimBlock, d_density, d_evenPsiHyper, dimensions, bodies, volume) << std::endl;

			double3 com = centerOfMass(h_evenPsiHyper, bsize, dxsize, dysize, dzsize, block_scale, d_p0);
			std::cout << "Center of mass: " << com.x << ", " << com.y << ", " << com.z << std::endl;
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
		itp_q << <dimGrid, edgeDimBlock >> > (d_oddQHyper, d_evenQHyper, d_evenPsiHyper, d_d0, dimensions, dt_per_sigma);
		itp_psi << <dimGrid, psiDimBlock >> > (d_oddPsiHyper, d_evenPsiHyper, d_evenQHyper, d_d1, d_hodges, dimensions, block_scale, d_p0, c0, c2, c4, dt);
		// Normalize
		normalize_h(dimGrid, psiDimBlock, d_density, d_oddPsiHyper, dimensions, bodies, volume);

		// Take an imaginary time step
		itp_q << <dimGrid, edgeDimBlock >> > (d_evenQHyper, d_oddQHyper, d_oddPsiHyper, d_d0, dimensions, dt_per_sigma);
		itp_psi << <dimGrid, psiDimBlock >> > (d_evenPsiHyper, d_oddPsiHyper, d_oddQHyper, d_d1, d_hodges, dimensions, block_scale, d_p0, c0, c2, c4, dt);
		// Normalize
		normalize_h(dimGrid, psiDimBlock, d_density, d_evenPsiHyper, dimensions, bodies, volume);

		iter++;
	}

#else
	std::string times = std::string("times = [times");
	std::string bqString = std::string("Bq = [Bq");
	std::string bbString = std::string("Bz = [Bz");
	std::string spinString = std::string("Spin = [Spin");
	std::string magX = std::string("mag_x = [mag_x");
	std::string magY = std::string("mag_y = [mag_y");
	std::string magZ = std::string("mag_z = [mag_z");
	std::string densityStr = std::string("norm = [norm");

	int lastSaveTime = 0;

#if HYPERBOLIC && !PARABOLIC
	std::string dirPrefix = "hyperbolic\\" + phaseToString(initPhase) + "\\" + getProjectionString() + "\\";
#elif PARABOLIC && !HYPERBOLIC
	std::string dirPrefix = "parabolic\\" + phaseToString(initPhase) + "\\" + getProjectionString() + "\\";
#else
	std::string dirPrefix = phaseToString(initPhase) + "\\" + getProjectionString() + "\\";
#endif

	std::string densDir = dirPrefix + "dens";
	std::string vtksDir = dirPrefix + "dens_vtks";
	std::string spinorVtksDir = dirPrefix + "spinor_vtks";
	std::string datsDir = dirPrefix + "dats";

	std::string createResultsDirCommand = "mkdir " + densDir;
	std::string createVtksDirCommand = "mkdir " + vtksDir;
	std::string createSpinorVtksDirCommand = "mkdir " + spinorVtksDir;
	std::string createDatsDirCommand = "mkdir " + datsDir;
	system(createResultsDirCommand.c_str());
	system(createVtksDirCommand.c_str());
	system(createSpinorVtksDirCommand.c_str());
	system(createDatsDirCommand.c_str());

	while (t < END_TIME)
	{
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
		std::cout << "Simulation time: " << t << " ms. Real time from previous save: " << duration.count() * 1e-9 << " s." << std::endl;
		prevTime = std::chrono::high_resolution_clock::now();

		// For checking the numerical stability
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

#if SAVE_PICTURE
#if HYPERBOLIC
		drawDensity(densDir, h_oddPsiHyper, dxsize, dysize, dzsize, t, Bs, d_p0, block_scale);
#endif
#if PARABOLIC
		drawDensity(densDir, h_oddPsiPara, dxsize, dysize, dzsize, t, Bs, d_p0, block_scale);
#endif
#endif

		// integrate one iteration
		for (uint step = 0; step < IMAGE_SAVE_FREQUENCY; step++)
		{
			// update odd values
			t += dt / omega_r * 1e3; // [ms]
			signal = getSignal(t);
			Bs.Bq = BqScale * signal.Bq;
			Bs.Bb = BzScale * signal.Bb;
			Bs.BqQuad = BqQuadScale * signal.Bq;
			Bs.BbQuad = BzQuadScale * signal.Bb;

#if HYPERBOLIC
			update_psi << <dimGrid, psiDimBlock >> > (d_oddPsiHyper, d_evenPsiHyper, d_evenQHyper, d_d1, d_hodges, Bs, dimensions, block_scale, d_p0, c0, c2, c4, dt, true);
			update_q << <dimGrid, edgeDimBlock >> > (d_oddQHyper, d_evenQHyper, d_evenPsiHyper, d_d0, dimensions, dt_per_sigma, true);
#endif
#if PARABOLIC
			update_psi << <dimGrid, psiDimBlock >> > (d_oddPsiPara, d_evenPsiPara, d_evenQPara, d_d1, d_hodges, Bs, dimensions, block_scale, d_p0, c0, c2, c4, dt, false);
			update_q << <dimGrid, edgeDimBlock >> > (d_oddQPara, d_oddQPara, d_oddPsiPara, d_d0, dimensions, dt_per_sigma, false);
#endif

			// update even values
			t += dt / omega_r * 1e3; // [ms]
			signal = getSignal(t);
			Bs.Bq = BqScale * signal.Bq;
			Bs.Bb = BzScale * signal.Bb;
			Bs.BqQuad = BqQuadScale * signal.Bq;
			Bs.BbQuad = BzQuadScale * signal.Bb;

#if HYPERBOLIC
			update_psi << <dimGrid, psiDimBlock >> > (d_evenPsiHyper, d_oddPsiHyper, d_oddQHyper, d_d1, d_hodges, Bs, dimensions, block_scale, d_p0, c0, c2, c4, dt, true);
			update_q << <dimGrid, edgeDimBlock >> > (d_evenQHyper, d_oddQHyper, d_oddPsiHyper, d_d0, dimensions, dt_per_sigma, true);
#endif
#if PARABOLIC
			update_psi << <dimGrid, psiDimBlock >> > (d_evenPsiPara, d_oddPsiPara, d_oddQPara, d_d1, d_hodges, Bs, dimensions, block_scale, d_p0, c0, c2, c4, dt, false);
			update_q << <dimGrid, edgeDimBlock >> > (d_evenQPara, d_evenQPara, d_evenPsiPara, d_d0, dimensions, dt_per_sigma, false);
#endif
		}
#if COMPUTE_ERROR
		// Compute error
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
		std::cout << hError << ", ";
#endif

#if SAVE_STATES
		// Copy back from device memory to host memory
#if HYPERBOLIC
		checkCudaErrors(cudaMemcpy3D(&oddPsiBackParamsHyper));
#endif
#if PARABOLIC
		checkCudaErrors(cudaMemcpy3D(&oddPsiBackParamsPara));
#endif
#if HYPERBOLIC
		if (t > 0.5)
		{
			saveVolume(vtksDir, h_oddPsiHyper, bsize, dxsize, dysize, dzsize, block_scale, d_p0, t);
			saveSpinor(spinorVtksDir, h_oddPsiHyper, bsize, dxsize, dysize, dzsize, block_scale, d_p0, t);
		}
#endif
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
	//checkCudaErrors(cudaFree(d_energy));
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

int main(int argc, char** argv)
{
	const double blockScale = DOMAIN_SIZE_X / REPLICABLE_STRUCTURE_COUNT_X / BLOCK_WIDTH_X;

	std::cout << "Start simulating from t = " << t << " ms, with a time step size of " << dt << "." << std::endl;
	std::cout << "The simulation will end at " << END_TIME << " ms." << std::endl;
	std::cout << "Block scale = " << blockScale << std::endl;
	std::cout << "Dual edge length = " << DUAL_EDGE_LENGTH * blockScale << std::endl;
	std::cout << "c0: " << c0 << ", c2: " << c2 << ", c4: " << c4 << std::endl;
#if USE_QUADRATIC_ZEEMAN
	std::cout << "Taking the quadratic Zeeman term into account" << std::endl;
#else
	std::cout << "No quadratic Zeeman term" << std::endl;
#endif
	std::cout << "Relativistic sigma = " << sigma << std::endl;

#if CREATE_KNOT
	std::cout << "Create knot" << std::endl;
#elif CREATE_SKYRMION
	std::cout << "Create skyrmion" << std::endl;
#endif
	printBasis();

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