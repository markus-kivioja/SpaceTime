#include <cuda_runtime.h>
#include "helper_cuda.h"

constexpr double CREATION_RAMP_START = 0.1;
constexpr double EXPANSION_START = CREATION_RAMP_START + 0.5; // When the expansion starts in ms

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
#include <cstdlib>

#include "mesh.h"

#define COMPUTE_GROUND_STATE 0

#define SAVE_STATES 0
#define SAVE_PICTURE 1

#define THREAD_BLOCK_X 16
#define THREAD_BLOCK_Y 2
#define THREAD_BLOCK_Z 1

bool USE_QUADRUPOLE_OFFSET = false;
bool USE_INITIAL_NOISE = false;

bool USE_QUADRATIC_ZEEMAN = false;
bool USE_THREE_BODY_LOSS = false;

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

const std::string GROUND_STATE_FILENAME = "ground_state.dat";
const std::string SAVE_FILE_PREFIX = "";

constexpr double NOISE_AMPLITUDE = 0.1;

double dt = 1e-4; // 1 x // Before the monopole creation ramp (0 - 200 ms)
//double dt = 1e-5; // 0.1 x // During and after the monopole creation ramp (200 ms - )

const double IMAGE_SAVE_INTERVAL = 0.5; // ms
uint IMAGE_SAVE_FREQUENCY = uint(IMAGE_SAVE_INTERVAL * 0.5 / 1e3 * omega_r / dt) + 1;

const uint STATE_SAVE_INTERVAL = 10.0; // ms

double t = 0; // Start time in ms
double END_TIME = 21.1; // End time in ms

double POLAR_FERRO_MIX = 0.5;

__device__ __inline__ double trap(double3 p, double t)
{
	if (t >= EXPANSION_START) {
		return 0;
	}

	double x = p.x * lambda_x;
	double y = p.y * lambda_y;
	double z = p.z * lambda_z;
	return 0.5 * (x * x + y * y + z * z) + 100.0;
}

__constant__ double quadrupoleCenterX = -0.20590789;
__constant__ double quadrupoleCenterY = -0.48902826;
__constant__ double quadrupoleCenterZ = -0.27353409;

__device__ __inline__ double3 magneticField(double3 p, double Bq, double3 Bb, bool USE_QUADRUPOLE_OFFSET)
{
	if (USE_QUADRUPOLE_OFFSET)
	{
		return {
			Bq * (p.x - quadrupoleCenterX) + Bb.x,
			Bq * (p.y - quadrupoleCenterY) + Bb.y,
			-2 * Bq * (p.z - quadrupoleCenterZ) + Bb.z
		};
	}
	else
	{
		return { Bq * p.x + Bb.x, Bq * p.y + Bb.y, -2 * Bq * p.z + Bb.z };
	}
}

__global__ void maxHamilton(double* maxHamlPtr, PitchedPtr prevStep, MagFields Bs, uint3 dimensions, double block_scale, double3 p0, double c0, double c2, double alpha, double t)
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
	const double totalPot = trap(globalPos, t) + c0 * normSq;

	double3 hamilton = { totalPot, totalPot, totalPot };

	const double2 temp = SQRT_2 * (conj(prev.s1) * prev.s0 + conj(prev.s0) * prev.s_1);
	const double3 magnetization = { temp.x, temp.y, normSq_s1 - normSq_s_1 };
	double3 B = magneticField(globalPos, Bs.Bq, Bs.Bb, false);
	B += c2 * magnetization;

	// Linear Zeeman shift
	hamilton.x += abs(INV_SQRT_2 * B.x);
	hamilton.y += abs(INV_SQRT_2 * B.y);
	hamilton.z += abs(B.z);

	size_t idx = zid * dimensions.x * dimensions.y * VALUES_IN_BLOCK + yid * dimensions.x * VALUES_IN_BLOCK + dataXid * VALUES_IN_BLOCK + dualNodeId;
	maxHamlPtr[idx] = max(hamilton.x, max(hamilton.y, hamilton.z));
};

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
	Complex3Vec psi = ((BlockPsis*)pPsi)->values[dualNodeId];

	size_t idx = VALUES_IN_BLOCK * (zid * dimensions.x * dimensions.y + yid * dimensions.x + dataXid) + dualNodeId;
	density[idx] = (psi.s1 * conj(psi.s1)).x + (psi.s0 * conj(psi.s0)).x + (psi.s_1 * conj(psi.s_1)).x;
}

__global__ void innerProduct(double* result, PitchedPtr pLeft, PitchedPtr pRight, uint3 dimensions)
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

	Complex3Vec left = ((BlockPsis*)(pLeft.ptr + pLeft.slicePitch * zid + pLeft.pitch * yid) + dataXid)->values[dualNodeId];
	Complex3Vec right = ((BlockPsis*)(pRight.ptr + pRight.slicePitch * zid + pRight.pitch * yid) + dataXid)->values[dualNodeId];

	size_t idx = VALUES_IN_BLOCK * (zid * dimensions.x * dimensions.y + yid * dimensions.x + dataXid) + dualNodeId;
	result[idx] = (conj(left.s1) * right.s1).x + (conj(left.s0) * right.s0).x + (conj(left.s_1) * right.s_1).x;
}

__global__ void localAvgSpinAndDensity(double* pSpinNorm, double3* pLocalAvgSpin, double* pDensity, PitchedPtr prevStep, uint3 dimensions)
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

	double density = normSq_s1 + normSq_s0 + normSq_s_1;

	psi.s1 = psi.s1 / sqrt(density);
	psi.s0 = psi.s0 / sqrt(density);
	psi.s_1 = psi.s_1 / sqrt(density);

	double2 temp = SQRT_2 * (conj(psi.s1) * psi.s0 + conj(psi.s0) * psi.s_1);
	double3 localAvgSpin = { temp.x, temp.y, normSq_s1 - normSq_s_1 };

	size_t idx = VALUES_IN_BLOCK * (zid * dimensions.x * dimensions.y + yid * dimensions.x + dataXid) + dualNodeId;

	pSpinNorm[idx] = density * sqrt(localAvgSpin.x * localAvgSpin.x + localAvgSpin.y * localAvgSpin.y + localAvgSpin.z * localAvgSpin.z);
	pLocalAvgSpin[idx] = localAvgSpin;
	pDensity[idx] = density;
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
	if (theta < 0) {
		theta += PI;
	}

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
	Complex3Vec psi = blockPsis->values[dualNodeId];
	double sqrtDens = sqrt(density[0]);
	psi.s1 = psi.s1 / sqrtDens;
	psi.s0 = psi.s0 / sqrtDens;
	psi.s_1 = psi.s_1 / sqrtDens;

	blockPsis->values[dualNodeId] = psi;
}

__global__ void polarState(PitchedPtr psi, const uint3 dimensions)
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
	pPsi->values[dualNodeId].s0 = { sqrt(normSq), 0 };
	pPsi->values[dualNodeId].s_1 = { 0, 0 };
};

__global__ void ferromagneticState(PitchedPtr psi, const uint3 dimensions)
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

	pPsi->values[dualNodeId].s1 = { sqrt(normSq), 0 };
	pPsi->values[dualNodeId].s0 = { 0, 0 };
	pPsi->values[dualNodeId].s_1 = { 0, 0 };
};

__global__ void mixedState(PitchedPtr psi, const uint3 dimensions, const double polarFerroMix)
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

	pPsi->values[dualNodeId].s1 = { sqrt(normSq * polarFerroMix), 0 };
	pPsi->values[dualNodeId].s0 = { sqrt(normSq * (1.0 - polarFerroMix)), 0 };
	pPsi->values[dualNodeId].s_1 = { 0, 0 };
};

#if COMPUTE_GROUND_STATE
__global__ void itp(PitchedPtr HPsiPtr, PitchedPtr nextStep, PitchedPtr prevStep, const int4* __restrict__ laplace, const double* __restrict__ hodges, MagFields Bs, const uint3 dimensions, const double block_scale, const double3 p0, const double c0, const double c2, const double dt, const double t)
{
	const size_t xid = blockIdx.x * blockDim.x + threadIdx.x;
	const size_t yid = blockIdx.y * blockDim.y + threadIdx.y;
	const size_t zid = blockIdx.z * blockDim.z + threadIdx.z;
	const size_t dataXid = xid / VALUES_IN_BLOCK; // One thread per every dual node so VALUES_IN_BLOCK threads per mesh block (on x-axis)
	const size_t dualNodeId = xid % VALUES_IN_BLOCK; // Dual node id. One thread per every dual node so VALUES_IN_BLOCK threads per mesh block (on x-axis)

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

	// For computing the energy/chemical potential
	BlockPsis* HPsi = (BlockPsis*)(HPsiPtr.ptr + HPsiPtr.slicePitch * zid + HPsiPtr.pitch * yid) + dataXid;

	// Update psi
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

		const double hodge = hodges[primaryFace] / (block_scale * block_scale);
		H.s1 += hodge * (otherBoundaryZeroCell.s1 - prev.s1);
		H.s0 += hodge * (otherBoundaryZeroCell.s0 - prev.s0);
		H.s_1 += hodge * (otherBoundaryZeroCell.s_1 - prev.s_1);

		primaryFace++;
	}

	const double normSq_s1 = prev.s1.x * prev.s1.x + prev.s1.y * prev.s1.y;
	const double normSq_s_1 = prev.s_1.x * prev.s_1.x + prev.s_1.y * prev.s_1.y;
	const double normSq = normSq_s1 + (prev.s0.x * prev.s0.x + prev.s0.y * prev.s0.y) + normSq_s_1;

	const double3 localPos = d_localPos[dualNodeId];
	const double3 globalPos = { p0.x + block_scale * (dataXid * BLOCK_WIDTH_X + localPos.x),
		p0.y + block_scale * (yid * BLOCK_WIDTH_Y + localPos.y),
		p0.z + block_scale * (zid * BLOCK_WIDTH_Z + localPos.z) };
	const double totalPot = trap(globalPos, t) + c0 * normSq;

	H.s1 += totalPot * prev.s1;
	H.s0 += totalPot * prev.s0;
	H.s_1 += totalPot * prev.s_1;

	const double2 magXY = SQRT_2 * (conj(prev.s1) * prev.s0 + conj(prev.s0) * prev.s_1);
	double3 B = c2 * double3{ magXY.x, magXY.y, normSq_s1 - normSq_s_1 };

	// Linear Zeeman shift
	const double2 Bxy = INV_SQRT_2 * double2{ B.x, B.y };
	const double2 BxyConj = conj(Bxy);
	H.s1 += (B.z * prev.s1 + BxyConj * prev.s0);
	H.s0 += (Bxy * prev.s1 + BxyConj * prev.s_1);
	H.s_1 += (Bxy * prev.s0 - B.z * prev.s_1);

	HPsi->values[dualNodeId].s1 = H.s1;
	HPsi->values[dualNodeId].s0 = H.s0;
	HPsi->values[dualNodeId].s_1 = H.s_1;

	nextPsi->values[dualNodeId].s1 = prev.s1 - dt * H.s1;
	nextPsi->values[dualNodeId].s0 = prev.s0 - dt * H.s0;
	nextPsi->values[dualNodeId].s_1 = prev.s_1 - dt * H.s_1;
};

__global__ void forwardEuler(PitchedPtr nextStep, PitchedPtr prevStep, int4* __restrict__ laplace, double* __restrict__ hodges, MagFields Bs, uint3 dimensions, double block_scale, double3 p0, double c0, double c2, double alpha, bool USE_THREE_BODY_LOSS, bool USE_QUADRATIC_ZEEMAN, bool USE_QUADRUPOLE_OFFSET, double dt, const double t)
{};
#else
__global__ void forwardEuler(PitchedPtr nextStep, PitchedPtr prevStep, int4* __restrict__ laplace, double* __restrict__ hodges, MagFields Bs, uint3 dimensions, double block_scale, double3 p0, double c0, double c2, double alpha, bool USE_THREE_BODY_LOSS, bool USE_QUADRATIC_ZEEMAN, bool USE_QUADRUPOLE_OFFSET, double dt, const double t)
{
	const size_t xid = blockIdx.x * blockDim.x + threadIdx.x;
	const size_t yid = blockIdx.y * blockDim.y + threadIdx.y;
	const size_t zid = blockIdx.z * blockDim.z + threadIdx.z;
	const size_t dataXid = xid / VALUES_IN_BLOCK; // One thread per every dual node so VALUES_IN_BLOCK threads per mesh block (on x-axis)
	const size_t dualNodeId = xid % VALUES_IN_BLOCK; // Dual node id. One thread per every dual node so VALUES_IN_BLOCK threads per mesh block (on x-axis)

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

		const double hodge = hodges[primaryFace] / (block_scale * block_scale);
		H.s1 += hodge * (otherBoundaryZeroCell.s1 - prev.s1);
		H.s0 += hodge * (otherBoundaryZeroCell.s0 - prev.s0);
		H.s_1 += hodge * (otherBoundaryZeroCell.s_1 - prev.s_1);

		primaryFace++;
	}

	const double normSq_s1 = prev.s1.x * prev.s1.x + prev.s1.y * prev.s1.y;
	const double normSq_s_1 = prev.s_1.x * prev.s_1.x + prev.s_1.y * prev.s_1.y;
	const double normSq = normSq_s1 + (prev.s0.x * prev.s0.x + prev.s0.y * prev.s0.y) + normSq_s_1;

	const double3 localPos = d_localPos[dualNodeId];
	const double3 globalPos = { p0.x + block_scale * (dataXid * BLOCK_WIDTH_X + localPos.x),
		p0.y + block_scale * (yid * BLOCK_WIDTH_Y + localPos.y),
		p0.z + block_scale * (zid * BLOCK_WIDTH_Z + localPos.z) };

	double2 totalPot = { trap(globalPos, t) + c0 * normSq, 0 };
	if (USE_THREE_BODY_LOSS)
	{
		totalPot.y = -alpha * normSq * normSq;
	}

	H.s1 += totalPot * prev.s1;
	H.s0 += totalPot * prev.s0;
	H.s_1 += totalPot * prev.s_1;

	const double2 magXY = SQRT_2 * (conj(prev.s1) * prev.s0 + conj(prev.s0) * prev.s_1);
	double3 B = magneticField(globalPos, Bs.Bq, Bs.Bb, USE_QUADRUPOLE_OFFSET);
	B += c2 * double3{ magXY.x, magXY.y, normSq_s1 - normSq_s_1 };

	// Linear Zeeman shift
	double2 Bxy = INV_SQRT_2 * double2{ B.x, B.y };
	double2 BxyConj = conj(Bxy);
	H.s1 += (B.z * prev.s1 + BxyConj * prev.s0);
	H.s0 += (Bxy * prev.s1 + BxyConj * prev.s_1);
	H.s_1 += (Bxy * prev.s0 - B.z * prev.s_1);

	if (USE_QUADRATIC_ZEEMAN)
	{
		// Quadratic Zeeman shift
		B = magneticField(globalPos, Bs.BqQuad, Bs.BbQuad, USE_QUADRUPOLE_OFFSET);
		Bxy = INV_SQRT_2 * double2{ B.x, B.y };
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
	}

	nextPsi->values[dualNodeId].s1 = prev.s1 + dt * double2{ H.s1.y, -H.s1.x };
	nextPsi->values[dualNodeId].s0 = prev.s0 + dt * double2{ H.s0.y, -H.s0.x };
	nextPsi->values[dualNodeId].s_1 = prev.s_1 + dt * double2{ H.s_1.y, -H.s_1.x };
};

__global__ void leapfrog(PitchedPtr nextStep, PitchedPtr prevStep, const int4* __restrict__ laplace, const double* __restrict__ hodges, MagFields Bs, const uint3 dimensions, const double block_scale, const double3 p0, const double c0, const double c2, double alpha, bool USE_THREE_BODY_LOSS, bool USE_QUADRATIC_ZEEMAN, bool USE_QUADRUPOLE_OFFSET, double dt, const double t)
{
	const size_t xid = blockIdx.x * blockDim.x + threadIdx.x;
	const size_t yid = blockIdx.y * blockDim.y + threadIdx.y;
	const size_t zid = blockIdx.z * blockDim.z + threadIdx.z;
	const size_t dataXid = xid / VALUES_IN_BLOCK; // One thread per every dual node so VALUES_IN_BLOCK threads per mesh block (on x-axis)
	const size_t dualNodeId = xid % VALUES_IN_BLOCK; // Dual node id. One thread per every dual node so VALUES_IN_BLOCK threads per mesh block (on x-axis)

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

		const double hodge = hodges[primaryFace] / (block_scale * block_scale);
		H.s1 += hodge * (otherBoundaryZeroCell.s1 - prev.s1);
		H.s0 += hodge * (otherBoundaryZeroCell.s0 - prev.s0);
		H.s_1 += hodge * (otherBoundaryZeroCell.s_1 - prev.s_1);

		primaryFace++;
	}

	const double normSq_s1 = prev.s1.x * prev.s1.x + prev.s1.y * prev.s1.y;
	const double normSq_s_1 = prev.s_1.x * prev.s_1.x + prev.s_1.y * prev.s_1.y;
	const double normSq = normSq_s1 + (prev.s0.x * prev.s0.x + prev.s0.y * prev.s0.y) + normSq_s_1;

	const double3 localPos = d_localPos[dualNodeId];
	const double3 globalPos = { p0.x + block_scale * (dataXid * BLOCK_WIDTH_X + localPos.x),
		p0.y + block_scale * (yid * BLOCK_WIDTH_Y + localPos.y),
		p0.z + block_scale * (zid * BLOCK_WIDTH_Z + localPos.z) };

	double2 totalPot = { trap(globalPos, t) + c0 * normSq, 0 };
	if (USE_THREE_BODY_LOSS)
	{
		totalPot.y = -alpha * normSq * normSq;
	}

	H.s1 += totalPot * prev.s1;
	H.s0 += totalPot * prev.s0;
	H.s_1 += totalPot * prev.s_1;

	const double2 magXY = SQRT_2 * (conj(prev.s1) * prev.s0 + conj(prev.s0) * prev.s_1);
	double3 B = magneticField(globalPos, Bs.Bq, Bs.Bb, USE_QUADRUPOLE_OFFSET);
	B += c2 * double3{ magXY.x, magXY.y, normSq_s1 - normSq_s_1 };

	// Linear Zeeman shift
	double2 Bxy = INV_SQRT_2 * double2{ B.x, B.y };
	double2 BxyConj = conj(Bxy);
	H.s1 += (B.z * prev.s1 + BxyConj * prev.s0);
	H.s0 += (Bxy * prev.s1 + BxyConj * prev.s_1);
	H.s_1 += (Bxy * prev.s0 - B.z * prev.s_1);

	if (USE_QUADRATIC_ZEEMAN)
	{
		// Quadratic Zeeman shift
		B = magneticField(globalPos, Bs.BqQuad, Bs.BbQuad, USE_QUADRUPOLE_OFFSET);
		Bxy = INV_SQRT_2 * double2{ B.x, B.y };
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
	}

	nextPsi->values[dualNodeId].s1 += 2 * dt * double2{ H.s1.y, -H.s1.x };
	nextPsi->values[dualNodeId].s0 += 2 * dt * double2{ H.s0.y, -H.s0.x };
	nextPsi->values[dualNodeId].s_1 += 2 * dt * double2{ H.s_1.y, -H.s_1.x };
};

__global__ void analyticStep(PitchedPtr nextStep, PitchedPtr prevStep, uint3 dimensions, const double2 phaseShift)
{
	const size_t xid = blockIdx.x * blockDim.x + threadIdx.x;
	const size_t yid = blockIdx.y * blockDim.y + threadIdx.y;
	const size_t zid = blockIdx.z * blockDim.z + threadIdx.z;
	const size_t dataXid = xid / VALUES_IN_BLOCK; // One thread per every dual node so VALUES_IN_BLOCK threads per mesh block (on x-axis)
	const size_t dualNodeId = xid % VALUES_IN_BLOCK; // Dual node id. One thread per every dual node so VALUES_IN_BLOCK threads per mesh block (on x-axis)

	// Exit leftover threads
	if (dataXid > dimensions.x || yid > dimensions.y || zid > dimensions.z)
	{
		return;
	}

	// Calculate the pointers for this block
	char* prevPsi = prevStep.ptr + prevStep.slicePitch * zid + prevStep.pitch * yid + sizeof(BlockPsis) * dataXid;
	BlockPsis* nextPsi = (BlockPsis*)(nextStep.ptr + nextStep.slicePitch * zid + nextStep.pitch * yid) + dataXid;

	const Complex3Vec prev = ((BlockPsis*)prevPsi)->values[dualNodeId];
	nextPsi->values[dualNodeId].s1 = prev.s1 * phaseShift;
	nextPsi->values[dualNodeId].s0 = prev.s0 * phaseShift;
	nextPsi->values[dualNodeId].s_1 = prev.s_1 * phaseShift;
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

double getMaxHamilton(dim3 dimGrid, dim3 dimBlock, double* maxHamlPtr, PitchedPtr psi, MagFields Bs, uint3 dimensions, size_t bodies, double block_scale, double3 p0)
{
	maxHamilton << <dimGrid, dimBlock >> > (maxHamlPtr, psi, Bs, dimensions, block_scale, p0, c0, c2, alpha, t);
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

	cudaPitchedPtr d_cudaEvenPsi;
	cudaPitchedPtr d_cudaOddPsi;
	checkCudaErrors(cudaMalloc3D(&d_cudaEvenPsi, psiExtent));
	checkCudaErrors(cudaMalloc3D(&d_cudaOddPsi, psiExtent));

	cudaPitchedPtr d_cudaHPsi;
	checkCudaErrors(cudaMalloc3D(&d_cudaHPsi, psiExtent));

	double* d_spinNorm;
	double* d_density;
	double* d_energy;
	double3* d_localAvgSpin;
	double3* d_u;
	double3* d_v;
	double* d_theta;
	checkCudaErrors(cudaMalloc(&d_spinNorm, bodies * sizeof(double)));
	checkCudaErrors(cudaMalloc(&d_density, bodies * sizeof(double)));
	checkCudaErrors(cudaMalloc(&d_energy, bodies * sizeof(double)));
	checkCudaErrors(cudaMalloc(&d_localAvgSpin, bodies * sizeof(double3)));
	checkCudaErrors(cudaMalloc(&d_u, bodies * sizeof(double3)));
	checkCudaErrors(cudaMalloc(&d_v, bodies * sizeof(double3)));
	checkCudaErrors(cudaMalloc(&d_theta, bodies * sizeof(double)));

	size_t offset = d_cudaEvenPsi.pitch * dysize + d_cudaEvenPsi.pitch + sizeof(BlockPsis);
	PitchedPtr d_evenPsi = { (char*)d_cudaEvenPsi.ptr + offset, d_cudaEvenPsi.pitch, d_cudaEvenPsi.pitch * dysize };
	PitchedPtr d_oddPsi = { (char*)d_cudaOddPsi.ptr + offset, d_cudaOddPsi.pitch, d_cudaOddPsi.pitch * dysize };

	PitchedPtr d_HPsi = { (char*)d_cudaHPsi.ptr + offset, d_cudaHPsi.pitch, d_cudaHPsi.pitch * dysize };

	// find terms for laplacian
	Buffer<int4> lapind;
	Buffer<double> hodges;
	getLaplacian(lapind, hodges, sizeof(BlockPsis), d_evenPsi.pitch, d_evenPsi.slicePitch);

	//std::cout << "lapsize = " << lapsize << ", lapfac = " << lapfac << ", lapfac0 = " << lapfac0 << std::endl;

	if (USE_QUADRUPOLE_OFFSET)
	{
		std::cout << "Quadrupole field offset is in use." << std::endl;
	}
	else
	{
		std::cout << "Not using quadrupole field offset." << std::endl;
	}

	for (int i = 0; i < hodges.size(); ++i) hodges[i] = -0.5 * hodges[i]; // / (block_scale * block_scale);

	int4* d_lapind;
	checkCudaErrors(cudaMalloc(&d_lapind, lapind.size() * sizeof(int4)));

	double* d_hodges;
	checkCudaErrors(cudaMalloc(&d_hodges, hodges.size() * sizeof(double)));

	// Initialize host memory
	size_t hostSize = dxsize * dysize * dzsize;
	BlockPsis* h_evenPsi;
	BlockPsis* h_oddPsi;
	checkCudaErrors(cudaMallocHost(&h_evenPsi, hostSize * sizeof(BlockPsis)));
	checkCudaErrors(cudaMallocHost(&h_oddPsi, hostSize * sizeof(BlockPsis)));
	memset(h_evenPsi, 0, hostSize * sizeof(BlockPsis));
	memset(h_oddPsi, 0, hostSize * sizeof(BlockPsis));

	double* h_density;
	double3* h_u;
	double* h_theta;
	double3* h_localAvgSpin;
	checkCudaErrors(cudaMallocHost(&h_density, bodies * sizeof(double)));
	checkCudaErrors(cudaMallocHost(&h_u, bodies * sizeof(double3)));
	checkCudaErrors(cudaMallocHost(&h_theta, bodies * sizeof(double)));
	checkCudaErrors(cudaMallocHost(&h_localAvgSpin, bodies * sizeof(double3)));

#if COMPUTE_GROUND_STATE
	// Initialize discrete field
	std::ifstream fs(GROUND_STATE_FILENAME, std::ios::binary | std::ios::in);
	if (fs.fail() != 0)
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
	else
	{
		std::cout << "Initialized ground state from file." << std::endl;

		fs.read((char*)&h_evenPsi[0], hostSize * sizeof(BlockPsis));
		fs.close();
	}

	bool loadGroundState = false;
	bool doForward = false;
#else
	bool loadGroundState = (t == 0);
	std::string filename = loadGroundState ? GROUND_STATE_FILENAME : SAVE_FILE_PREFIX + toString(t) + ".dat";
	std::ifstream fs(filename, std::ios::binary | std::ios::in);
	if (fs.fail() != 0)
	{
		std::cout << "Failed to open file " << filename << std::endl;
		return 1;
	}
	fs.read((char*)&h_oddPsi[0], hostSize * sizeof(BlockPsis));
	fs.close();

	if (USE_THREE_BODY_LOSS)
	{
		std::cout << "The three-body loss is taken into account." << std::endl;
	}
	else
	{
		std::cout << "The three-body loss is ignored." << std::endl;
	}

	if (USE_QUADRATIC_ZEEMAN)
	{
		std::cout << "The quadratic Zeeman shift is taken into account." << std::endl;
	}
	else
	{
		std::cout << "The quadratic Zeeman shift is ignored." << std::endl;
	}

	if (USE_INITIAL_NOISE)
	{
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
							const double2 rand_s1 = { distribution(generator), distribution(generator) };
							const double2 rand_s0 = { distribution(generator), distribution(generator) };
							const double2 rand_s_1 = { distribution(generator), distribution(generator) };

							const double dens_s1 = (conj(h_oddPsi[dstI].values[l].s1) * h_oddPsi[dstI].values[l].s1).x;
							const double dens_s0 = (conj(h_oddPsi[dstI].values[l].s0) * h_oddPsi[dstI].values[l].s0).x;
							const double dens_s_1 = (conj(h_oddPsi[dstI].values[l].s_1) * h_oddPsi[dstI].values[l].s_1).x;
							const double dens = dens_s1 + dens_s0 + dens_s_1;

							h_oddPsi[dstI].values[l].s1 += sqrt(dens) * NOISE_AMPLITUDE * rand_s1;
							h_oddPsi[dstI].values[l].s0 += sqrt(dens) * NOISE_AMPLITUDE * rand_s0;
							h_oddPsi[dstI].values[l].s_1 += sqrt(dens) * NOISE_AMPLITUDE * rand_s_1;
						}
					}
				}
			}
			std::cout << "Initial noise of " << NOISE_AMPLITUDE << " applied." << std::endl;
		}
	}
	else
	{
		std::cout << "No initial noise." << std::endl;
	}

	bool doForward = true;
	std::string evenFilename = SAVE_FILE_PREFIX + "even_" + toString(t) + ".dat";
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
	checkCudaErrors(cudaMemcpy(d_lapind, &lapind[0], lapind.size() * sizeof(int4), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_hodges, &hodges[0], hodges.size() * sizeof(double), cudaMemcpyHostToDevice));

	// Clear host memory after data has been copied to devices
	cudaDeviceSynchronize();
	lapind.clear();
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
	dim3 dimBlock(THREAD_BLOCK_X * VALUES_IN_BLOCK, THREAD_BLOCK_Y, THREAD_BLOCK_Z);
	dim3 dimGrid((xsize + THREAD_BLOCK_X - 1) / THREAD_BLOCK_X,
		(ysize + THREAD_BLOCK_Y - 1) / THREAD_BLOCK_Y,
		((zsize + THREAD_BLOCK_Z - 1) / THREAD_BLOCK_Z));

	Signal signal;
	MagFields Bs{};

	const double volume = block_scale * block_scale * block_scale * VOLUME;

	if (loadGroundState)
	{
		if (USE_INITIAL_NOISE)
		{
			normalize_h(dimGrid, dimBlock, d_density, d_oddPsi, dimensions, bodies, volume);
			std::cout << "Density after normilizing the noised ground state:" << std::endl;
			printDensity(dimGrid, dimBlock, d_density, d_oddPsi, dimensions, bodies, volume);
		}
		if (POLAR_FERRO_MIX == 0.0)
		{
			std::cout << "Transform ground state to polar phase" << std::endl;
			polarState << <dimGrid, dimBlock >> > (d_oddPsi, dimensions);
		}
		else if (POLAR_FERRO_MIX == 1.0)
		{
			std::cout << "Transform ground state to ferromagnetic phase" << std::endl;
			ferromagneticState << <dimGrid, dimBlock >> > (d_oddPsi, dimensions);
		}
		else
		{
			std::cout << "Transform ground state to mixed phase with a mix of " << POLAR_FERRO_MIX << std::endl;
			mixedState << <dimGrid, dimBlock >> > (d_oddPsi, dimensions, POLAR_FERRO_MIX);
		}

		printDensity(dimGrid, dimBlock, d_density, d_oddPsi, dimensions, bodies, volume);
	}

	const double E = 127.295; // Computed with ITP
	double analytic_t = 0;

	// Take one forward Euler step if starting from the ground state or time step changed
	if (doForward)
	{
		std::cout << "No even time step file found. Doing one forward step." << std::endl;

		signal = getSignal(t);
		Bs.Bq = BqScale * signal.Bq;
		Bs.Bb = BzScale * signal.Bb;
		Bs.BqQuad = BqQuadScale * signal.Bq;
		Bs.BbQuad = BzQuadScale * signal.Bb;
		forwardEuler << <dimGrid, dimBlock >> > (d_evenPsi, d_oddPsi, d_lapind, d_hodges, Bs, dimensions, block_scale, d_p0, c0, c2, alpha, USE_THREE_BODY_LOSS, USE_QUADRATIC_ZEEMAN, USE_QUADRUPOLE_OFFSET, dt, t);
		//analyticStep << <dimGrid, dimBlock >> > (d_evenPsi, d_oddPsi, dimensions, double2{ cos(-analytic_t * E), sin(-analytic_t * E) });
		//analytic_t += dt;
	}
	else
	{
		std::cout << "Skipping the forward step." << std::endl;
	}

#if COMPUTE_GROUND_STATE
	uint iter = 0;

	normalize_h(dimGrid, dimBlock, d_density, d_evenPsi, dimensions, bodies, volume);

	while (true)
	{
		if ((iter % 1000) == 0) std::cout << "Iteration " << iter << std::endl;
#if SAVE_PICTURE
		if ((iter % 1000) == 0)
		{
			checkCudaErrors(cudaMemcpy3D(&evenPsiBackParams));
			drawDensity("GS", h_evenPsi, dxsize, dysize, dzsize, iter, "ground_state");
			printDensity(dimGrid, dimBlock, d_density, d_evenPsi, dimensions, bodies, volume);

			double3 com = centerOfMass(h_evenPsi, bsize, dxsize, dysize, dzsize, block_scale, d_p0);
			std::cout << "Center of mass: " << com.x << ", " << com.y << ", " << com.z << std::endl;
		}
#endif
		if (iter == 100000)
		{
			//polarState<<<dimGrid, dimBlock>>>(d_evenPsi, dimensions);

			checkCudaErrors(cudaMemcpy3D(&evenPsiBackParams));
			std::ofstream fs(GROUND_STATE_FILENAME, std::ios::binary | std::ios_base::trunc);
			if (fs.fail() != 0) return 1;
			fs.write((char*)&h_evenPsi[0], hostSize * sizeof(BlockPsis));
			fs.close();
			return 0;
		}
		// Take an imaginary time step
		itp << <dimGrid, dimBlock >> > (d_HPsi, d_oddPsi, d_evenPsi, d_lapind, d_hodges, Bs, dimensions, block_scale, d_p0, c0, c2, dt, 0);
		// Normalize
		normalize_h(dimGrid, dimBlock, d_density, d_oddPsi, dimensions, bodies, volume);

		// Take an imaginary time step
		itp << <dimGrid, dimBlock >> > (d_HPsi, d_evenPsi, d_oddPsi, d_lapind, d_hodges, Bs, dimensions, block_scale, d_p0, c0, c2, dt, 0);
		// Normalize
		normalize_h(dimGrid, dimBlock, d_density, d_evenPsi, dimensions, bodies, volume);

		// Compute energy/chemical potential // Alice ring experimen E = 127.295
		innerProduct << <dimGrid, dimBlock >> > (d_energy, d_evenPsi, d_HPsi, dimensions);
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

	std::string resultsDir = getProjectionString() + "\\results_" + std::to_string(POLAR_FERRO_MIX);
	std::string vtksDir = getProjectionString() + "\\vtks_" + std::to_string(POLAR_FERRO_MIX);
	std::string datsDir = getProjectionString() + "\\dats_" + std::to_string(POLAR_FERRO_MIX);

	std::string createResultsDirCommand = "mkdir " + resultsDir;
	std::string createVtksDirCommand = "mkdir " + vtksDir;
	std::string createDatsDirCommand = "mkdir " + datsDir;
	system(createResultsDirCommand.c_str());
	system(createVtksDirCommand.c_str());
	system(createDatsDirCommand.c_str());

	double expansionBlockScale = block_scale;

	// Measure wall clock time
	static auto prevTime = std::chrono::high_resolution_clock::now();

	while (t < CREATION_RAMP_START)
	{
		// update odd values
		t += dt / omega_r * 1e3; // [ms]
		signal = getSignal(t);
		Bs.Bq = BqScale * signal.Bq;
		Bs.Bb = BzScale * signal.Bb;
		Bs.BqQuad = BqQuadScale * signal.Bq;
		Bs.BbQuad = BzQuadScale * signal.Bb;
		leapfrog << <dimGrid, dimBlock >> > (d_oddPsi, d_evenPsi, d_lapind, d_hodges, Bs, dimensions, expansionBlockScale, d_p0, c0, c2, alpha, USE_THREE_BODY_LOSS, USE_QUADRATIC_ZEEMAN, USE_QUADRUPOLE_OFFSET, dt, t);

		// update even values
		t += dt / omega_r * 1e3; // [ms]
		signal = getSignal(t);
		Bs.Bq = BqScale * signal.Bq;
		Bs.Bb = BzScale * signal.Bb;
		Bs.BqQuad = BqQuadScale * signal.Bq;
		Bs.BbQuad = BzQuadScale * signal.Bb;
		leapfrog << <dimGrid, dimBlock >> > (d_evenPsi, d_oddPsi, d_lapind, d_hodges, Bs, dimensions, expansionBlockScale, d_p0, c0, c2, alpha, USE_THREE_BODY_LOSS, USE_QUADRATIC_ZEEMAN, USE_QUADRUPOLE_OFFSET, dt, t);
	}

#if SAVE_PICTURE
	// Copy back from device memory to host memory
	checkCudaErrors(cudaMemcpy3D(&oddPsiBackParams));

	// Measure wall clock time
	auto duration = std::chrono::high_resolution_clock::now() - prevTime;
	std::cout << "Simulation time: " << t << " ms. Real time from previous save: " << duration.count() * 1e-9 << " s." << std::endl;
	prevTime = std::chrono::high_resolution_clock::now();

	drawDensity("", h_oddPsi, dxsize, dysize, dzsize, t - CREATION_RAMP_START, resultsDir);

	//uvTheta << <dimGrid, dimBlock >> > (d_u, d_v, d_theta, d_oddPsi, dimensions);
	//cudaMemcpy(h_u, d_u, bodies * sizeof(double3), cudaMemcpyDeviceToHost);
	//cudaMemcpy(h_theta, d_theta, bodies * sizeof(double), cudaMemcpyDeviceToHost);
	//drawUtheta(h_u, h_theta, xsize, ysize, zsize, t - 202.03);
	//
	//ferromagneticDomain << <dimGrid, dimBlock >> > (d_ferroDom, d_oddPsi, dimensions);
	//cudaMemcpy(h_ferroDom, d_ferroDom, bodies * sizeof(double), cudaMemcpyDeviceToHost);
	//drawFerroDom(h_ferroDom, xsize, ysize, zsize, t - 202.03);
#endif

	while (t < END_TIME)
	{
		const uint centerIdx = 57 * dxsize * dysize + 57 * dxsize + 57;
		double2 temp = h_oddPsi[centerIdx].values[5].s0;
		double startPhase = atan2(temp.y, temp.x);
		double phaseTime = 0;

		// integrate one iteration
		for (uint step = 0; step < IMAGE_SAVE_FREQUENCY; step++)
		{
			// update odd values
			phaseTime += dt;
			t += dt / omega_r * 1e3; // [ms]
			if (t >= EXPANSION_START) {
				double k = 0.82; // 0.7569772335291065; // From the Aalto QCD code for F=2
				expansionBlockScale += dt / omega_r * 1e3 * k * block_scale;
			}
			signal = getSignal(t);
			Bs.Bq = BqScale * signal.Bq;
			Bs.Bb = BzScale * signal.Bb;
			Bs.BqQuad = BqQuadScale * signal.Bq;
			Bs.BbQuad = BzQuadScale * signal.Bb;
			leapfrog << <dimGrid, dimBlock >> > (d_oddPsi, d_evenPsi, d_lapind, d_hodges, Bs, dimensions, expansionBlockScale, d_p0, c0, c2, alpha, USE_THREE_BODY_LOSS, USE_QUADRATIC_ZEEMAN, USE_QUADRUPOLE_OFFSET, dt, t);

			// update even values
			phaseTime += dt;
			t += dt / omega_r * 1e3; // [ms]
			if (t >= EXPANSION_START) {
				double k = 0.82; // 0.7569772335291065; // From the Aalto QCD code for F=2
				expansionBlockScale += dt / omega_r * 1e3 * k * block_scale;
			}
			signal = getSignal(t);
			Bs.Bq = BqScale * signal.Bq;
			Bs.Bb = BzScale * signal.Bb;
			Bs.BqQuad = BqQuadScale * signal.Bq;
			Bs.BbQuad = BzQuadScale * signal.Bb;
			leapfrog << <dimGrid, dimBlock >> > (d_evenPsi, d_oddPsi, d_lapind, d_hodges, Bs, dimensions, expansionBlockScale, d_p0, c0, c2, alpha, USE_THREE_BODY_LOSS, USE_QUADRATIC_ZEEMAN, USE_QUADRUPOLE_OFFSET, dt, t);
		}

#if SAVE_PICTURE
		// Copy back from device memory to host memory
		checkCudaErrors(cudaMemcpy3D(&oddPsiBackParams));

		temp = h_oddPsi[centerIdx].values[5].s0;
		double endPhase = atan2(temp.y, temp.x);
		double phaseDiff = endPhase - startPhase;
		std::cout << "Energy was " << phaseDiff / phaseTime << std::endl;

		// Measure wall clock time
		auto duration = std::chrono::high_resolution_clock::now() - prevTime;
		std::cout << "Simulation time: " << t << " ms. Real time from previous save: " << duration.count() * 1e-9 << " s." << std::endl;
		prevTime = std::chrono::high_resolution_clock::now();

		drawDensity("", h_oddPsi, dxsize, dysize, dzsize, t - CREATION_RAMP_START, resultsDir);

		//uvTheta << <dimGrid, dimBlock >> > (d_u, d_v, d_theta, d_oddPsi, dimensions);
		//cudaMemcpy(h_u, d_u, bodies * sizeof(double3), cudaMemcpyDeviceToHost);
		//cudaMemcpy(h_theta, d_theta, bodies * sizeof(double), cudaMemcpyDeviceToHost);
		//drawUtheta(h_u, h_theta, xsize, ysize, zsize, t - 202.03);
		//
		//ferromagneticDomain << <dimGrid, dimBlock >> > (d_ferroDom, d_oddPsi, dimensions);
		//cudaMemcpy(h_ferroDom, d_ferroDom, bodies * sizeof(double), cudaMemcpyDeviceToHost);
		//drawFerroDom(h_ferroDom, xsize, ysize, zsize, t - 202.03);
#endif
#if SAVE_STATES
		// Copy back from device memory to host memory
		checkCudaErrors(cudaMemcpy3D(&oddPsiBackParams));

		localAvgSpinAndDensity << <dimGrid, dimBlock >> > (d_spinNorm, d_localAvgSpin, d_density, d_oddPsi, dimensions);
		cudaMemcpy(h_localAvgSpin, d_localAvgSpin, bodies * sizeof(double3), cudaMemcpyDeviceToHost);
		uvTheta << <dimGrid, dimBlock >> > (d_u, d_v, d_theta, d_oddPsi, dimensions);
		cudaMemcpy(h_u, d_u, bodies * sizeof(double3), cudaMemcpyDeviceToHost);
		cudaMemcpy(h_theta, d_theta, bodies * sizeof(double), cudaMemcpyDeviceToHost);
		if (t > 19.9)
			saveVolume(SAVE_FILE_PREFIX, h_oddPsi, h_localAvgSpin, h_u, h_theta, bsize, dxsize, dysize, dzsize, 0, block_scale, d_p0, t, vtksDir);

		SpinMagDens spinMagDens = integrateSpinAndDensity(dimGrid, dimBlock, d_spinNorm, d_localAvgSpin, d_density, bodies, volume);
		times += ", " + toString(t);
		bqString += ", " + toString(Bs.Bq);
		bzString += ", " + toString(Bs.Bb.x + Bs.Bb.y + Bs.Bb.z);
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
			textFile.save(datsDir + "/" + SAVE_FILE_PREFIX + toString(t) + ".m");

			std::ofstream oddFs(datsDir + "/" + SAVE_FILE_PREFIX + toString(t) + ".dat", std::ios::binary | std::ios_base::trunc);
			if (oddFs.fail() != 0) return 1;
			oddFs.write((char*)&h_oddPsi[0], hostSize * sizeof(BlockPsis));
			oddFs.close();

			checkCudaErrors(cudaMemcpy3D(&evenPsiBackParams));
			std::ofstream evenFs(datsDir + "/" + SAVE_FILE_PREFIX + "even_" + toString(t) + ".dat", std::ios::binary | std::ios_base::trunc);
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

	checkCudaErrors(cudaFree(d_cudaEvenPsi.ptr));
	checkCudaErrors(cudaFree(d_cudaOddPsi.ptr));
	checkCudaErrors(cudaFree(d_spinNorm));
	checkCudaErrors(cudaFree(d_density));
	checkCudaErrors(cudaFree(d_localAvgSpin));
	checkCudaErrors(cudaFree(d_u));
	checkCudaErrors(cudaFree(d_v));
	checkCudaErrors(cudaFree(d_theta));
	checkCudaErrors(cudaFree(d_lapind));
	checkCudaErrors(cudaFree(d_hodges));
	checkCudaErrors(cudaFreeHost(h_evenPsi));
	checkCudaErrors(cudaFreeHost(h_oddPsi));
	checkCudaErrors(cudaFreeHost(h_density));
	checkCudaErrors(cudaFreeHost(h_u));
	checkCudaErrors(cudaFreeHost(h_theta));
	checkCudaErrors(cudaFreeHost(h_localAvgSpin));

	return 0;
}

double sigma = 0.01;
double dt_per_sigma = dt / sigma;

void readConfFile(const std::string& confFileName)
{
	std::ifstream file;
	file.open(confFileName, std::ios::in);
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
			else if (size_t pos = line.find("qz") != std::string::npos)
			{
				USE_QUADRATIC_ZEEMAN = true;
			}
			else if (size_t pos = line.find("offset") != std::string::npos)
			{
				USE_QUADRUPOLE_OFFSET = true;
			}
			else if (size_t pos = line.find("noise") != std::string::npos)
			{
				USE_INITIAL_NOISE = true;
			}
			else if (size_t pos = line.find("loss") != std::string::npos)
			{
				USE_THREE_BODY_LOSS = true;
			}
			else if (size_t pos = line.find("pol_fer") != std::string::npos)
			{
				POLAR_FERRO_MIX = std::stod(line.substr(pos + 7));
			}
			//else if (size_t pos = line.find("expand") != std::string::npos)
			//{
			//	EXPANSION_START = std::stod(line.substr(pos + 6));
			//}
		}
	}
}

int main(int argc, char** argv)
{
	if (argc > 1)
	{
		std::cout << "Read config " << argv[1] << std::endl;
		readConfFile(std::string(argv[1]));
	}

	const double targetBlockWidth = DOMAIN_SIZE_X / REPLICABLE_STRUCTURE_COUNT_X;
	const double blockScale = targetBlockWidth / BLOCK_WIDTH_X;

	std::cout << "Start simulating from t = " << t << " ms, with a time step size of " << dt << "." << std::endl;
	std::cout << "The simulation will end at " << END_TIME << " ms." << std::endl;
	//std::cout << "Block scale = " << blockScale << std::endl;
	//std::cout << "Dual edge length = " << DUAL_EDGE_LENGTH * blockScale << std::endl;
	std::cout << "Image save interval is " << IMAGE_SAVE_INTERVAL << " ms." << std::endl;
	std::cout << "Mix betweem polar and ferromagnetic phases = " << POLAR_FERRO_MIX << std::endl;
	printBasis();

	// integrate in time using DEC
	auto domainMin = Vector3(-DOMAIN_SIZE_X * 0.5, -DOMAIN_SIZE_Y * 0.5, -DOMAIN_SIZE_Z * 0.5);
	auto domainMax = Vector3(DOMAIN_SIZE_X * 0.5, DOMAIN_SIZE_Y * 0.5, DOMAIN_SIZE_Z * 0.5);
	//for (POLAR_FERRO_MIX = 0.0; POLAR_FERRO_MIX <= 1.0; POLAR_FERRO_MIX += 0.1)
	{
		//t = 0;
		integrateInTime(blockScale, domainMin, domainMax);
	}

	return 0;
}
