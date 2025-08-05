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
Phase initPhase = Phase::BN_VERT;

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

Phase stringToPhase(const std::string& phase)
{
	if (phase == "un") return Phase::UN;
	if (phase == "bn_vert") return Phase::BN_VERT;
	if (phase == "bn_hori") return Phase::BN_HORI;
	if (phase == "cyclic") return Phase::CYCLIC;
	return Phase::UN;
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

#include "utils.h"

// Experimental field ramps from D.S. Hall (Amherst)
constexpr myFloat STATE_PREP_DURATION = 0.1;
constexpr myFloat CREATION_RAMP_DURATION = 0.0177;
constexpr myFloat HOLD_TIME = 0.25; // 0.5;
constexpr myFloat HOLD_TIME_EXTRA_DELAY = 0.005;
constexpr myFloat TOTAL_HOLD_TIME = HOLD_TIME + HOLD_TIME_EXTRA_DELAY;
constexpr myFloat PROJECTION_RAMP_DURATION = 0.120;
constexpr myFloat OPT_TRAP_OFF_DELAY = 0.020;
constexpr myFloat OPT_TRAP_OFF = STATE_PREP_DURATION + CREATION_RAMP_DURATION + TOTAL_HOLD_TIME + OPT_TRAP_OFF_DELAY; // When the expansion starts in ms
constexpr myFloat GRADIENT_OFF_DELAY = 0.010;
constexpr myFloat GRADIENT_OFF_DUARATION = 0.034;
constexpr myFloat GRID_SCALING_START = 1.0; // ms

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

#define COMPUTE_GROUND_STATE 1
#define GROUND_STATE_ITERATION_COUNT 100000

#define USE_QUADRATIC_ZEEMAN 0
#define USE_QUADRUPOLE_OFFSET 0
#define USE_INITIAL_NOISE 0

#define SAVE_STATES 0
#define SAVE_PICTURE 1

#define THREAD_BLOCK_X 16
#define THREAD_BLOCK_Y 2
#define THREAD_BLOCK_Z 1

constexpr myFloat DOMAIN_SIZE_X = 20.0; //24.0;
constexpr myFloat DOMAIN_SIZE_Y = 20.0; //24.0;
constexpr myFloat DOMAIN_SIZE_Z = 20.0; //24.0;

constexpr myFloat REPLICABLE_STRUCTURE_COUNT_X = 112.0;
//constexpr myFloat REPLICABLE_STRUCTURE_COUNT_Y = 112.0;
//constexpr myFloat REPLICABLE_STRUCTURE_COUNT_Z = 112.0;

//constexpr myFloat k = 0.7569772335291065; // Grid upscale speed for expansion (from QCD code)
constexpr myFloat k = 1.0; // Grid upscale speed for expansion (from own experiments)

constexpr myFloat N = 2e5; // Number of atoms in the condensate

constexpr myFloat trapFreq_r = 126;
constexpr myFloat trapFreq_z = 166;

constexpr myFloat omega_r = trapFreq_r * 2 * PI;
constexpr myFloat omega_z = trapFreq_z * 2 * PI;
constexpr myFloat lambda_x = 1.0;
constexpr myFloat lambda_y = 1.0;
constexpr myFloat lambda_z = omega_z / omega_r;

constexpr myFloat a_bohr = 5.2917721092e-11; //[m] Bohr radius
constexpr myFloat a_0 = 87.9;
constexpr myFloat a_2 = 91.41;
constexpr myFloat a_4 = 98.36;

constexpr myFloat atomMass = 1.44316060e-25;
constexpr myFloat hbar = 1.05457148e-34; // [m^2 kg / s]
const myFloat a_r = sqrt(hbar / (atomMass * omega_r)); //[m]

const myFloat c0 = 4 * PI * N * (4 * a_2 + 3 * a_4) * a_bohr / (7 * a_r);
const myFloat c2 = 4 * PI * N * (a_4 - a_2) * a_bohr / (7 * a_r);
const myFloat c4 = 4 * PI * N * (7 * a_0 - 10 * a_2 + 3 * a_4) * a_bohr / (7 * a_r);

constexpr myFloat myGamma = 2.9e-30;
//const myFloat alpha = N * N * myGamma * 1e-12 / (a_r * a_r * a_r * a_r * a_r * a_r * 2 * PI * trapFreq_r);
const myFloat alpha = 0;

constexpr myFloat muB = 9.27400968e-24; // [m^2 kg / s^2 T^-1] Bohr magneton

const myFloat BqScale = -(0.5 * muB / (hbar * omega_r) * a_r) / 100.; // [cm/Gauss]
constexpr myFloat BzScale = -(0.5 * muB / (hbar * omega_r)) / 10000.; // [1/Gauss]

constexpr myFloat A_hfs = 3.41734130545215;
const myFloat BqQuadScale = 100 * a_r * sqrt(0.25 * 1000 * (1.399624624 * 1.399624624) / (trapFreq_r * 2 * A_hfs)); //[cm/Gauss]
const myFloat BzQuadScale = sqrt(0.25 * 1000 * (1.399624624 * 1.399624624) / (trapFreq_r * 2 * A_hfs)); //[1/Gauss]  \sqrt{g_q}

constexpr myFloat SQRT_2 = 1.41421356237309;
//constexpr myFloat INV_SQRT_2 = 0.70710678118655;

std::string toStringShort(const myFloat value)
{
	std::ostringstream out;
	out.precision(2);
	out << std::fixed << value;
	return out.str();
};

const std::string GROUND_STATE_FILENAME = "ground_state_psi_" + toStringShort(DOMAIN_SIZE_X) + "_" + toStringShort(REPLICABLE_STRUCTURE_COUNT_X) + ".dat";
constexpr myFloat NOISE_AMPLITUDE = 0; //0.1;

//constexpr myFloat dt = 1e-4; // 1 x // Before the monopole creation ramp (0 - 200 ms)
constexpr myFloat dt = 1e-5; // 0.1 x // During and after the monopole creation ramp (200 ms - )

const myFloat IMAGE_SAVE_INTERVAL = 0.25; // ms
const uint IMAGE_SAVE_FREQUENCY = uint(IMAGE_SAVE_INTERVAL * 0.5 / 1e3 * omega_r / dt) + 1;

const uint STATE_SAVE_INTERVAL = 10.0; // ms

myFloat t = 0; // Start time in ms
constexpr myFloat END_TIME = OPT_TRAP_OFF + GRADIENT_OFF_DELAY + GRADIENT_OFF_DUARATION + 24.0; // End time in ms

myFloat relativePhase = 0; // 5.105088062083414; // In radians

__host__ __device__ __inline__ myFloat trap(myFloat3 p, myFloat t)
{
	if (t >= OPT_TRAP_OFF) {
		return 0;
	}

	myFloat x = p.x * lambda_x;
	myFloat y = p.y * lambda_y;
	myFloat z = p.z * lambda_z;
	return 0.5 * (x * x + y * y + z * z) + 100.0;
}

__constant__ myFloat quadrupoleCenterX = -0.20590789;
__constant__ myFloat quadrupoleCenterY = -0.48902826;
__constant__ myFloat quadrupoleCenterZ = -0.27353409;

__device__ __inline__ myFloat3 magneticField(myFloat3 p, myFloat Bq, myFloat3 Bb)
{
	return { Bq * p.x + Bb.x, Bq * p.y + Bb.y, -2 * Bq * p.z + Bb.z };
}

__global__ void density(myFloat* density, PitchedPtr prevStep, uint3 dimensions)
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

__global__ void innerProduct(myFloat* result, PitchedPtr pLeft, PitchedPtr pRight, uint3 dimensions)
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

	size_t idx = VALUES_IN_BLOCK * (zid * dimensions.x * dimensions.y + yid * dimensions.x + dataXid) + dualNodeId;
	result[idx] = (conj(left.s2) * right.s2).x + (conj(left.s1) * right.s1).x + (conj(left.s0) * right.s0).x + (conj(left.s_1) * right.s_1).x + (conj(left.s_2) * right.s_2).x;
}

__global__ void localAvgSpinAndDensity(myFloat* pSpinNorm, myFloat3* pLocalAvgSpin, myFloat* pDensity, PitchedPtr prevStep, uint3 dimensions)
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

	myFloat normSq_s2 = psi.s2.x * psi.s2.x + psi.s2.y * psi.s2.y;
	myFloat normSq_s1 = psi.s1.x * psi.s1.x + psi.s1.y * psi.s1.y;
	myFloat normSq_s0 = psi.s0.x * psi.s0.x + psi.s0.y * psi.s0.y;
	myFloat normSq_s_1 = psi.s_1.x * psi.s_1.x + psi.s_1.y * psi.s_1.y;
	myFloat normSq_s_2 = psi.s_2.x * psi.s_2.x + psi.s_2.y * psi.s_2.y;

	myFloat density = normSq_s2 + normSq_s1 + normSq_s0 + normSq_s_1 + normSq_s_2;

	psi.s2 = psi.s2 / sqrt(density);
	psi.s1 = psi.s1 / sqrt(density);
	psi.s0 = psi.s0 / sqrt(density);
	psi.s_1 = psi.s_1 / sqrt(density);
	psi.s_2 = psi.s_2 / sqrt(density);

	myFloat2 temp = SQRT_2 * (conj(psi.s1) * psi.s0 + conj(psi.s0) * psi.s_1);
	myFloat3 localAvgSpin = { temp.x, temp.y, normSq_s1 - normSq_s_1 };

	size_t idx = VALUES_IN_BLOCK * (zid * dimensions.x * dimensions.y + yid * dimensions.x + dataXid) + dualNodeId;

	pSpinNorm[idx] = density * sqrt(localAvgSpin.x * localAvgSpin.x + localAvgSpin.y * localAvgSpin.y + localAvgSpin.z * localAvgSpin.z);
	pLocalAvgSpin[idx] = localAvgSpin;
	pDensity[idx] = density;
}

__global__ void uvTheta(myFloat3* out_u, myFloat3* out_v, myFloat* outTheta, PitchedPtr psiPtr, uint3 dimensions)
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
	Complex5Vec psi = ((BlockPsis*)pPsi)->values[dualNodeId];

	// a = m + in
	myFloat2 ax = (psi.s_1 - psi.s1) / SQRT_2;
	myFloat2 ay = myFloat2{ 0, -1 } *(psi.s_1 + psi.s1) / SQRT_2;
	myFloat2 az = psi.s0;
	myFloat3 m = myFloat3{ ax.x, ay.x, az.x };
	myFloat3 n = myFloat3{ ax.y, ay.y, az.y };

	myFloat m_dot_n = m.x * n.x + m.y * n.y + m.z * n.z;
	myFloat mNormSqr = m.x * m.x + m.y * m.y + m.z * m.z;
	myFloat nNormSqr = n.x * n.x + n.y * n.y + n.z * n.z;

	myFloat theta = atan2(-2 * m_dot_n, mNormSqr - nNormSqr) / 2;

	myFloat sinTheta = sin(theta);
	myFloat cosTheta = cos(theta);
	myFloat3 u = myFloat3{ m.x * cosTheta - sinTheta * n.x, m.y * cosTheta - sinTheta * n.y, m.z * cosTheta - sinTheta * n.z };
	myFloat3 v = myFloat3{ m.x * sinTheta + cosTheta * n.x, m.y * sinTheta + cosTheta * n.y, m.z * sinTheta + cosTheta * n.z };
	myFloat uNorm = sqrt(u.x * u.x + u.y * u.y + u.z * u.z);
	myFloat vNorm = sqrt(v.x * v.x + v.y * v.y + v.z * v.z);

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

__global__ void integrate(myFloat* dataVec, size_t stride, bool addLast, myFloat dv)
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

__global__ void integrateVec(myFloat3* dataVec, size_t stride, bool addLast, myFloat dv)
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

__global__ void integrateVecWithDensity(myFloat3* dataVec, myFloat* density, size_t stride, bool addLast, myFloat dv)
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

__global__ void reduceMax(myFloat* dataVec, size_t stride, bool addLast)
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

__global__ void normalize(myFloat* density, PitchedPtr psiPtr, uint3 dimensions)
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
	myFloat sqrtDens = sqrt(density[0]);
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

	myFloat normSq_s2 = prev.s2.x * prev.s2.x + prev.s2.y * prev.s2.y;
	myFloat normSq_s1 = prev.s1.x * prev.s1.x + prev.s1.y * prev.s1.y;
	myFloat normSq_s0 = prev.s0.x * prev.s0.x + prev.s0.y * prev.s0.y;
	myFloat normSq_s_1 = prev.s_1.x * prev.s_1.x + prev.s_1.y * prev.s_1.y;
	myFloat normSq_s_2 = prev.s_2.x * prev.s_2.x + prev.s_2.y * prev.s_2.y;
	myFloat normSq = normSq_s2 + normSq_s1 + normSq_s0 + normSq_s_1 + normSq_s_2;

	pPsi->values[dualNodeId].s2 = { 0, 0 };
	pPsi->values[dualNodeId].s1 = { 0, 0 };
	pPsi->values[dualNodeId].s0 = { sqrt(normSq), 0 };
	pPsi->values[dualNodeId].s_1 = { 0, 0 };
	pPsi->values[dualNodeId].s_2 = { 0, 0 };
};

__global__ void horizontalBnState(PitchedPtr psi, uint3 dimensions, myFloat phase = 0) // Horizontal orientation of the biaxial nematic phase
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

	myFloat normSq_s2 = prev.s2.x * prev.s2.x + prev.s2.y * prev.s2.y;
	myFloat normSq_s1 = prev.s1.x * prev.s1.x + prev.s1.y * prev.s1.y;
	myFloat normSq_s0 = prev.s0.x * prev.s0.x + prev.s0.y * prev.s0.y;
	myFloat normSq_s_1 = prev.s_1.x * prev.s_1.x + prev.s_1.y * prev.s_1.y;
	myFloat normSq_s_2 = prev.s_2.x * prev.s_2.x + prev.s_2.y * prev.s_2.y;
	myFloat normSq = normSq_s2 + normSq_s1 + normSq_s0 + normSq_s_1 + normSq_s_2;

	myFloat amplitude = sqrt(normSq / 2);

	pPsi->values[dualNodeId].s2 = { amplitude, 0 };
	pPsi->values[dualNodeId].s1 = { 0, 0 };
	pPsi->values[dualNodeId].s0 = { 0, 0 };
	pPsi->values[dualNodeId].s_1 = { 0, 0 };
	pPsi->values[dualNodeId].s_2 = { cos(phase) * amplitude, sin(phase) * amplitude };
};

__global__ void verticalBnState(PitchedPtr psi, uint3 dimensions, myFloat phase = 0) // Vertical orientation of the biaxial nematic phase
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

	myFloat normSq_s2 = prev.s2.x * prev.s2.x + prev.s2.y * prev.s2.y;
	myFloat normSq_s1 = prev.s1.x * prev.s1.x + prev.s1.y * prev.s1.y;
	myFloat normSq_s0 = prev.s0.x * prev.s0.x + prev.s0.y * prev.s0.y;
	myFloat normSq_s_1 = prev.s_1.x * prev.s_1.x + prev.s_1.y * prev.s_1.y;
	myFloat normSq_s_2 = prev.s_2.x * prev.s_2.x + prev.s_2.y * prev.s_2.y;
	myFloat normSq = normSq_s2 + normSq_s1 + normSq_s0 + normSq_s_1 + normSq_s_2;

	myFloat amplitude = sqrt(normSq / 2);

	pPsi->values[dualNodeId].s2 = { 0, 0 };
	pPsi->values[dualNodeId].s1 = { amplitude, 0 };
	pPsi->values[dualNodeId].s0 = { 0, 0 };
	pPsi->values[dualNodeId].s_1 = { cos(phase) * amplitude, sin(phase) * amplitude };
	pPsi->values[dualNodeId].s_2 = { 0, 0 };
};

__global__ void cyclicState(PitchedPtr psi, uint3 dimensions, myFloat phase = 0) // Cyclic phase
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

	myFloat normSq_s2 = prev.s2.x * prev.s2.x + prev.s2.y * prev.s2.y;
	myFloat normSq_s1 = prev.s1.x * prev.s1.x + prev.s1.y * prev.s1.y;
	myFloat normSq_s0 = prev.s0.x * prev.s0.x + prev.s0.y * prev.s0.y;
	myFloat normSq_s_1 = prev.s_1.x * prev.s_1.x + prev.s_1.y * prev.s_1.y;
	myFloat normSq_s_2 = prev.s_2.x * prev.s_2.x + prev.s_2.y * prev.s_2.y;
	myFloat normSq = normSq_s2 + normSq_s1 + normSq_s0 + normSq_s_1 + normSq_s_2;

	myFloat amplitude_m2 = sqrt(normSq * 1 / 3);
	myFloat amplitude_m_1 = sqrt(normSq * 2 / 3);

	pPsi->values[dualNodeId].s2 = { amplitude_m2, 0 };
	pPsi->values[dualNodeId].s1 = { 0, 0 };
	pPsi->values[dualNodeId].s0 = { 0, 0 };
	pPsi->values[dualNodeId].s_1 = { cos(phase) * amplitude_m_1, sin(phase) * amplitude_m_1 };
	pPsi->values[dualNodeId].s_2 = { 0, 0 };
};

#if COMPUTE_GROUND_STATE
__global__ void itp(PitchedPtr HPsiPtr, PitchedPtr nextStep, PitchedPtr prevStep, const int4* __restrict__ laplace, const myFloat* __restrict__ hodges, MagFields Bs, const uint3 dimensions, const myFloat block_scale, const myFloat3 p0, const myFloat c0, const myFloat c2, const myFloat c4, myFloat t)
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
	const Complex5Vec prev = ((BlockPsis*)prevPsi)->values[dualNodeId];
	ldsPrevPsis[threadIdxInBlock].values[dualNodeId] = prev;

	// Kill also the leftover edge threads
	if (dataXid == dimensions.x || yid == dimensions.y || zid == dimensions.z)
	{
		return;
	}
	__syncthreads();

	uint primaryFace = dualNodeId * FACE_COUNT;

	Complex5Vec H;
	H.s2 = { 0, 0 };
	H.s1 = { 0, 0 };
	H.s0 = { 0, 0 };
	H.s_1 = { 0, 0 };
	H.s_2 = { 0, 0 };

	// Add the Laplacian to the Hamiltonian
#pragma unroll
	for (int i = 0; i < FACE_COUNT; ++i)
	{
		const int4 laplacian = laplace[primaryFace];

		const int neighbourX = localDataXid + laplacian.x;
		const int neighbourY = threadIdx.y + laplacian.y;
		const int neighbourZ = threadIdx.z + laplacian.z;

		Complex5Vec otherBoundaryZeroCell;
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

		const myFloat hodge = hodges[primaryFace] / (block_scale * block_scale);
		H.s2 += hodge * (otherBoundaryZeroCell.s2 - prev.s2);
		H.s1 += hodge * (otherBoundaryZeroCell.s1 - prev.s1);
		H.s0 += hodge * (otherBoundaryZeroCell.s0 - prev.s0);
		H.s_1 += hodge * (otherBoundaryZeroCell.s_1 - prev.s_1);
		H.s_2 += hodge * (otherBoundaryZeroCell.s_2 - prev.s_2);

		primaryFace++;
	}

	const myFloat normSq_s2 = prev.s2.x * prev.s2.x + prev.s2.y * prev.s2.y;
	const myFloat normSq_s1 = prev.s1.x * prev.s1.x + prev.s1.y * prev.s1.y;
	const myFloat normSq_s0 = prev.s0.x * prev.s0.x + prev.s0.y * prev.s0.y;
	const myFloat normSq_s_1 = prev.s_1.x * prev.s_1.x + prev.s_1.y * prev.s_1.y;
	const myFloat normSq_s_2 = prev.s_2.x * prev.s_2.x + prev.s_2.y * prev.s_2.y;
	const myFloat normSq = normSq_s2 + normSq_s1 + normSq_s0 + normSq_s_1 + normSq_s_2;

	const myFloat Fz = c2 * (2.0 * normSq_s2 + normSq_s1 - normSq_s_1 - 2.0 * normSq_s_2);

	const myFloat3 localPos = d_localPos[dualNodeId];
	const myFloat3 globalPos = { p0.x + block_scale * (dataXid * BLOCK_WIDTH_X + localPos.x),
		p0.y + block_scale * (yid * BLOCK_WIDTH_Y + localPos.y),
		p0.z + block_scale * (zid * BLOCK_WIDTH_Z + localPos.z) };
	//const myFloat totalPot = trap(globalPos, t) + c0 * normSq;
	myFloat2 ab = { trap(globalPos, t) + c0 * normSq, 0 };

	myFloat3 B = { 0 }; //magneticField(globalPos, Bs.Bq, Bs.Bz);

	Complex5Vec diagonalTerm;
	diagonalTerm.s2 = myFloat2{ 2.0f * Fz + 0.4f * c4 * normSq_s_2 - 2.0f * B.z, 0 } + ab;
	diagonalTerm.s1 = myFloat2{ Fz + 0.4f * c4 * normSq_s_1 - B.z, 0 } + ab;
	diagonalTerm.s0 = myFloat2{ 0.2f * c4 * normSq_s0, 0 } + ab;
	diagonalTerm.s_1 = myFloat2{ -Fz + 0.4f * c4 * normSq_s1 + B.z, 0 } + ab;
	diagonalTerm.s_2 = myFloat2{ -2.0f * Fz + 0.4f * c4 * normSq_s2 + 2.0f * B.z, 0 } + ab;

	H.s2 += diagonalTerm.s2 * prev.s2;    // psi1
	H.s1 += diagonalTerm.s1 * prev.s1;    // psi2
	H.s0 += diagonalTerm.s0 * prev.s0;    // psi3
	H.s_1 += diagonalTerm.s_1 * prev.s_1; // psi4
	H.s_2 += diagonalTerm.s_2 * prev.s_2; // psi5

	myFloat2 denominator = c2 * (2.0 * (prev.s2 * conj(prev.s1) +
		prev.s_1 * conj(prev.s_2)) +
		sqrt(6.0) * (prev.s1 * conj(prev.s0) +
			prev.s0 * conj(prev.s_1))) - myFloat2{ B.x, -B.y };

	myFloat2 c12 = denominator - 0.4 * c4 * prev.s_1 * conj(prev.s_2);
	myFloat2 c45 = denominator - 0.4 * c4 * prev.s2 * conj(prev.s1);
	myFloat2 c13 = 0.2 * c4 * prev.s0 * conj(prev.s_2);
	myFloat2 c35 = 0.2 * c4 * prev.s2 * conj(prev.s0);
	myFloat2 c23 = sqrt(1.5) * denominator - 0.2 * c4 * prev.s0 * conj(prev.s_1);
	myFloat2 c34 = sqrt(1.5) * denominator - 0.2 * c4 * prev.s1 * conj(prev.s0);

	H.s2  += (c12 * prev.s1 + c13 * prev.s0);
	H.s1  += (conj(c12) * prev.s2 + c23 * prev.s0);
	H.s0  += (conj(c13) * prev.s2 + c35 * prev.s_2 + c34 * prev.s_1 + conj(c23) * prev.s1);
	H.s_1 += (conj(c34) * prev.s0 + c45 * prev.s_2);
	H.s_2 += (conj(c35) * prev.s0 + conj(c45) * prev.s_1);

	HPsi->values[dualNodeId].s2 = H.s2;
	HPsi->values[dualNodeId].s1 = H.s1;
	HPsi->values[dualNodeId].s0 = H.s0;
	HPsi->values[dualNodeId].s_1 = H.s_1;
	HPsi->values[dualNodeId].s_2 = H.s_2;

	nextPsi->values[dualNodeId].s2 = prev.s2 - dt * H.s2;
	nextPsi->values[dualNodeId].s1 = prev.s1 - dt * H.s1;
	nextPsi->values[dualNodeId].s0 = prev.s0 - dt * H.s0;
	nextPsi->values[dualNodeId].s_1 = prev.s_1 - dt * H.s_1;
	nextPsi->values[dualNodeId].s_2 = prev.s_2 - dt * H.s_2;
};

__global__ void forwardEuler(PitchedPtr nextStep, PitchedPtr prevStep, int4* __restrict__ laplace, myFloat* __restrict__ hodges, MagFields Bs, uint3 dimensions, myFloat block_scale, myFloat3 p0, myFloat c0, myFloat c2, myFloat c4, myFloat alpha, myFloat t)
{};
#else
__global__ void forwardEuler(PitchedPtr nextStep, PitchedPtr prevStep, int4* __restrict__ laplace, myFloat* __restrict__ hodges, MagFields Bs, uint3 dimensions, myFloat block_scale, myFloat3 p0, myFloat c0, myFloat c2, myFloat c4, myFloat alpha, myFloat t)
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
	const Complex5Vec prev = ((BlockPsis*)prevPsi)->values[dualNodeId];
	ldsPrevPsis[threadIdxInBlock].values[dualNodeId] = prev;

	// Kill also the leftover edge threads
	if (dataXid == dimensions.x || yid == dimensions.y || zid == dimensions.z)
	{
		return;
	}
	__syncthreads();

	uint primaryFace = dualNodeId * FACE_COUNT;

	Complex5Vec H;
	H.s2 = { 0, 0 };
	H.s1 = { 0, 0 };
	H.s0 = { 0, 0 };
	H.s_1 = { 0, 0 };
	H.s_2 = { 0, 0 };

	// Add the Laplacian to the Hamiltonian
#pragma unroll
	for (int i = 0; i < FACE_COUNT; ++i)
	{
		const int4 laplacian = laplace[primaryFace];

		const int neighbourX = localDataXid + laplacian.x;
		const int neighbourY = threadIdx.y + laplacian.y;
		const int neighbourZ = threadIdx.z + laplacian.z;

		Complex5Vec otherBoundaryZeroCell;
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

		const myFloat hodge = hodges[primaryFace] / (block_scale * block_scale);
		H.s2 += hodge * (otherBoundaryZeroCell.s2 - prev.s2);
		H.s1 += hodge * (otherBoundaryZeroCell.s1 - prev.s1);
		H.s0 += hodge * (otherBoundaryZeroCell.s0 - prev.s0);
		H.s_1 += hodge * (otherBoundaryZeroCell.s_1 - prev.s_1);
		H.s_2 += hodge * (otherBoundaryZeroCell.s_2 - prev.s_2);

		primaryFace++;
	}

	const myFloat normSq_s2 = prev.s2.x * prev.s2.x + prev.s2.y * prev.s2.y;
	const myFloat normSq_s1 = prev.s1.x * prev.s1.x + prev.s1.y * prev.s1.y;
	const myFloat normSq_s0 = prev.s0.x * prev.s0.x + prev.s0.y * prev.s0.y;
	const myFloat normSq_s_1 = prev.s_1.x * prev.s_1.x + prev.s_1.y * prev.s_1.y;
	const myFloat normSq_s_2 = prev.s_2.x * prev.s_2.x + prev.s_2.y * prev.s_2.y;
	const myFloat normSq = normSq_s2 + normSq_s1 + normSq_s0 + normSq_s_1 + normSq_s_2;

	const myFloat Fz = c2 * (2.0 * normSq_s2 + normSq_s1 - normSq_s_1 - 2.0 * normSq_s_2);

	const myFloat3 localPos = d_localPos[dualNodeId];
	const myFloat3 globalPos = { p0.x + block_scale * (dataXid * BLOCK_WIDTH_X + localPos.x),
		p0.y + block_scale * (yid * BLOCK_WIDTH_Y + localPos.y),
		p0.z + block_scale * (zid * BLOCK_WIDTH_Z + localPos.z) };
	//const myFloat totalPot = trap(globalPos, t) + c0 * normSq;
	myFloat2 ab = { trap(globalPos, t) + c0 * normSq, -alpha * normSq * normSq };

	myFloat3 B = magneticField(globalPos, Bs.Bq, Bs.Bb);

	Complex5Vec diagonalTerm;
	diagonalTerm.s2 = myFloat2{ 2.0f * Fz + 0.4f * c4 * normSq_s_2 - 2.0f * B.z, 0 } + ab;
	diagonalTerm.s1 = myFloat2{ Fz + 0.4f * c4 * normSq_s_1 - B.z, 0 } + ab;
	diagonalTerm.s0 = myFloat2{ 0.2f * c4 * normSq_s0, 0 } + ab;
	diagonalTerm.s_1 = myFloat2{ -Fz + 0.4f * c4 * normSq_s1 + B.z, 0 } + ab;
	diagonalTerm.s_2 = myFloat2{ -2.0f * Fz + 0.4f * c4 * normSq_s2 + 2.0f * B.z, 0 } + ab;

	H.s2 += diagonalTerm.s2 * prev.s2;    // psi1
	H.s1 += diagonalTerm.s1 * prev.s1;    // psi2
	H.s0 += diagonalTerm.s0 * prev.s0;    // psi3
	H.s_1 += diagonalTerm.s_1 * prev.s_1; // psi4
	H.s_2 += diagonalTerm.s_2 * prev.s_2; // psi5

	myFloat2 denominator = c2 * (2.0 * (prev.s2 * conj(prev.s1) +
		prev.s_1 * conj(prev.s_2)) +
		sqrt(6.0) * (prev.s1 * conj(prev.s0) +
			prev.s0 * conj(prev.s_1))) - myFloat2{ B.x, -B.y };

	myFloat2 c12 = denominator - 0.4 * c4 * prev.s_1 * conj(prev.s_2);
	myFloat2 c45 = denominator - 0.4 * c4 * prev.s2 * conj(prev.s1);
	myFloat2 c13 = 0.2 * c4 * prev.s0 * conj(prev.s_2);
	myFloat2 c35 = 0.2 * c4 * prev.s2 * conj(prev.s0);
	myFloat2 c23 = sqrt(1.5) * denominator - 0.2 * c4 * prev.s0 * conj(prev.s_1);
	myFloat2 c34 = sqrt(1.5) * denominator - 0.2 * c4 * prev.s1 * conj(prev.s0);

	H.s2  += (c12 * prev.s1 + c13 * prev.s0);
	H.s1  += (conj(c12) * prev.s2 + c23 * prev.s0);
	H.s0  += (conj(c13) * prev.s2 + c35 * prev.s_2 + c34 * prev.s_1 + conj(c23) * prev.s1);
	H.s_1 += (conj(c34) * prev.s0 + c45 * prev.s_2);
	H.s_2 += (conj(c35) * prev.s0 + conj(c45) * prev.s_1);

#if USE_QUADRATIC_ZEEMAN
	B = magneticField(globalPos, Bs.BqQuad, Bs.BbQuad);
	const myFloat c = sqrt(6.0) / 2.0;
	const myFloat2 Bxy = { B.x, B.y };
	const myFloat Bz = B.z;
	const myFloat BxyNormSq = (conj(Bxy) * Bxy).x;
	H.s2  -= (4 * Bz * Bz + BxyNormSq) * prev.s2 + (3 * Bz * conj(Bxy)) * prev.s1 + (c * conj(Bxy) * conj(Bxy)) * prev.s0 + (0) * prev.s_1 + (0) * prev.s_2;
	H.s1  -= (3 * Bz * Bxy) * prev.s2 + (Bz * Bz + (5 / 2) * BxyNormSq) * prev.s1 + (Bz * c * conj(Bxy)) * prev.s0 + ((3 / 2) * conj(Bxy) * conj(Bxy)) * prev.s_1 + (0) * prev.s_2;
	H.s0  -= (c * Bxy * Bxy) * prev.s2 + (c * Bz * Bxy) * prev.s1 + (3 * BxyNormSq) * prev.s0 + (-Bz * c * conj(Bxy)) * prev.s_1 + (c * conj(Bxy) * conj(Bxy)) * prev.s_2;
	H.s_1 -= (0) * prev.s2 + ((3 / 2) * Bxy * Bxy) * prev.s1 + (-Bz * c * Bxy) * prev.s0 + ((5 / 2) * BxyNormSq + Bz * Bz) * prev.s_1 + (-3 * Bz * conj(Bxy)) * prev.s_2;
	H.s_2 -= (0) * prev.s2 + (0) * prev.s1 + (c * Bxy * Bxy) * prev.s0 + (-3 * Bz * Bxy) * prev.s_1 + (BxyNormSq + 4 * Bz * Bz) * prev.s_2;
#endif

	nextPsi->values[dualNodeId].s2 = prev.s2 + dt * myFloat2{ H.s2.y, -H.s2.x };
	nextPsi->values[dualNodeId].s1 = prev.s1 + dt * myFloat2{ H.s1.y, -H.s1.x };
	nextPsi->values[dualNodeId].s0 = prev.s0 + dt * myFloat2{ H.s0.y, -H.s0.x };
	nextPsi->values[dualNodeId].s_1 = prev.s_1 + dt * myFloat2{ H.s_1.y, -H.s_1.x };
	nextPsi->values[dualNodeId].s_2 = prev.s_2 + dt * myFloat2{ H.s_2.y, -H.s_2.x };
};

__global__ void leapfrog(PitchedPtr nextStep, PitchedPtr prevStep, const int4* __restrict__ laplace, const myFloat* __restrict__ hodges, MagFields Bs, const uint3 dimensions, const myFloat block_scale, const myFloat3 p0, const myFloat c0, const myFloat c2, const myFloat c4, myFloat alpha, myFloat t)
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
	const Complex5Vec prev = ((BlockPsis*)prevPsi)->values[dualNodeId];
	ldsPrevPsis[threadIdxInBlock].values[dualNodeId] = prev;

	// Kill also the leftover edge threads
	if (dataXid == dimensions.x || yid == dimensions.y || zid == dimensions.z)
	{
		return;
	}
	__syncthreads();

	uint primaryFace = dualNodeId * FACE_COUNT;

	Complex5Vec H;
	H.s2 = { 0, 0 };
	H.s1 = { 0, 0 };
	H.s0 = { 0, 0 };
	H.s_1 = { 0, 0 };
	H.s_2 = { 0, 0 };

	// Add the Laplacian to the Hamiltonian
#pragma unroll
	for (int i = 0; i < FACE_COUNT; ++i)
	{
		const int4 laplacian = laplace[primaryFace];

		const int neighbourX = localDataXid + laplacian.x;
		const int neighbourY = threadIdx.y + laplacian.y;
		const int neighbourZ = threadIdx.z + laplacian.z;

		Complex5Vec otherBoundaryZeroCell;
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

		const myFloat hodge = hodges[primaryFace] / (block_scale * block_scale);
		H.s2 += hodge * (otherBoundaryZeroCell.s2 - prev.s2);
		H.s1 += hodge * (otherBoundaryZeroCell.s1 - prev.s1);
		H.s0 += hodge * (otherBoundaryZeroCell.s0 - prev.s0);
		H.s_1 += hodge * (otherBoundaryZeroCell.s_1 - prev.s_1);
		H.s_2 += hodge * (otherBoundaryZeroCell.s_2 - prev.s_2);

		primaryFace++;
	}

	const myFloat normSq_s2 = prev.s2.x * prev.s2.x + prev.s2.y * prev.s2.y;
	const myFloat normSq_s1 = prev.s1.x * prev.s1.x + prev.s1.y * prev.s1.y;
	const myFloat normSq_s0 = prev.s0.x * prev.s0.x + prev.s0.y * prev.s0.y;
	const myFloat normSq_s_1 = prev.s_1.x * prev.s_1.x + prev.s_1.y * prev.s_1.y;
	const myFloat normSq_s_2 = prev.s_2.x * prev.s_2.x + prev.s_2.y * prev.s_2.y;
	const myFloat normSq = normSq_s2 + normSq_s1 + normSq_s0 + normSq_s_1 + normSq_s_2;

	const myFloat3 localPos = d_localPos[dualNodeId];
	const myFloat3 globalPos = { p0.x + block_scale * (dataXid * BLOCK_WIDTH_X + localPos.x),
		p0.y + block_scale * (yid * BLOCK_WIDTH_Y + localPos.y),
		p0.z + block_scale * (zid * BLOCK_WIDTH_Z + localPos.z) };

	myFloat2 ab = { trap(globalPos, t) + c0 * normSq, -alpha * normSq * normSq };

	myFloat3 B = magneticField(globalPos, Bs.Bq, Bs.Bb);

	const myFloat Fz = c2 * (2.0 * normSq_s2 + normSq_s1 - normSq_s_1 - 2.0 * normSq_s_2);

	Complex5Vec diagonalTerm;
	diagonalTerm.s2 = myFloat2{  2.0f * Fz + 0.4f * c4 * normSq_s_2 - 2.0f * B.z, 0 } + ab;
	diagonalTerm.s1 = myFloat2{        Fz + 0.4f * c4 * normSq_s_1 -       B.z, 0 } + ab;
	diagonalTerm.s0 = myFloat2{             0.2f * c4 * normSq_s0             , 0 } + ab;
	diagonalTerm.s_1 = myFloat2{      -Fz + 0.4f * c4 * normSq_s1  +       B.z, 0 } + ab;
	diagonalTerm.s_2 = myFloat2{-2.0f * Fz + 0.4f * c4 * normSq_s2  + 2.0f * B.z, 0 } + ab;

	H.s2 += diagonalTerm.s2 * prev.s2;    // psi1
	H.s1 += diagonalTerm.s1 * prev.s1;    // psi2
	H.s0 += diagonalTerm.s0 * prev.s0;    // psi3
	H.s_1 += diagonalTerm.s_1 * prev.s_1; // psi4
	H.s_2 += diagonalTerm.s_2 * prev.s_2; // psi5

	myFloat2 denominator = c2 * (2.0 * (prev.s2  * conj(prev.s1) +
		                               prev.s_1 * conj(prev.s_2)) +
		                  sqrt(6.0) * (prev.s1  * conj(prev.s0) +
			                           prev.s0  * conj(prev.s_1))) - myFloat2{ B.x, -B.y };

	myFloat2 c12 = denominator - 0.4 * c4 * prev.s_1 * conj(prev.s_2);
	myFloat2 c45 = denominator - 0.4 * c4 * prev.s2 * conj(prev.s1);
	myFloat2 c13 = 0.2 * c4 * prev.s0 * conj(prev.s_2);
	myFloat2 c35 = 0.2 * c4 * prev.s2 * conj(prev.s0);
	myFloat2 c23 = sqrt(1.5) * denominator - 0.2 * c4 * prev.s0 * conj(prev.s_1);
	myFloat2 c34 = sqrt(1.5) * denominator - 0.2 * c4 * prev.s1 * conj(prev.s0);

	H.s2  += (c12 * prev.s1 + c13 * prev.s0);
	H.s1  += (conj(c12) * prev.s2 + c23 * prev.s0);
	H.s0  += (conj(c13) * prev.s2 + c35 * prev.s_2 + c34 * prev.s_1 + conj(c23) * prev.s1);
	H.s_1 += (conj(c34) * prev.s0 + c45 * prev.s_2);
	H.s_2 += (conj(c35) * prev.s0 + conj(c45) * prev.s_1);

#if USE_QUADRATIC_ZEEMAN
	B = magneticField(globalPos, Bs.BqQuad, Bs.BbQuad);
	const myFloat c = sqrt(6.0) / 2.0;
	const myFloat2 Bxy = { B.x, B.y };
	const myFloat Bz = B.z;
	const myFloat BxyNormSq = (conj(Bxy) * Bxy).x;
	H.s2  -= (4 * Bz * Bz + BxyNormSq) * prev.s2 + (3 * Bz * conj(Bxy)) * prev.s1 + (c * conj(Bxy) * conj(Bxy)) * prev.s0 + (0) * prev.s_1 + (0) * prev.s_2;
	H.s1  -= (3 * Bz * Bxy) * prev.s2 + (Bz * Bz + (5 / 2) * BxyNormSq) * prev.s1 + (Bz * c* conj(Bxy)) * prev.s0 + ((3 / 2) * conj(Bxy) * conj(Bxy)) * prev.s_1 + (0) * prev.s_2;
	H.s0  -= (c * Bxy * Bxy) * prev.s2 + (c * Bz* Bxy) * prev.s1 + (3 * BxyNormSq) * prev.s0 + (-Bz * c * conj(Bxy)) * prev.s_1 + (c * conj(Bxy) * conj(Bxy)) * prev.s_2;
	H.s_1 -= (0) * prev.s2 + ((3 / 2) * Bxy * Bxy) * prev.s1 + (-Bz * c * Bxy) * prev.s0 + ((5 / 2) * BxyNormSq + Bz * Bz) * prev.s_1 + (-3 * Bz * conj(Bxy)) * prev.s_2;
	H.s_2 -= (0) * prev.s2 + (0) * prev.s1 + (c * Bxy * Bxy) * prev.s0 + (-3 * Bz * Bxy) * prev.s_1 + (BxyNormSq + 4 * Bz * Bz) * prev.s_2;
#endif

	nextPsi->values[dualNodeId].s2 += 2 * dt * myFloat2{ H.s2.y, -H.s2.x };
	nextPsi->values[dualNodeId].s1 += 2 * dt * myFloat2{ H.s1.y, -H.s1.x };
	nextPsi->values[dualNodeId].s0 += 2 * dt * myFloat2{ H.s0.y, -H.s0.x };
	nextPsi->values[dualNodeId].s_1 += 2 * dt * myFloat2{ H.s_1.y, -H.s_1.x };
	nextPsi->values[dualNodeId].s_2 += 2 * dt * myFloat2{ H.s_2.y, -H.s_2.x };
};
#endif

__device__ 	myFloat3 getGlobalPos(int blockX, int blockY, int blockZ, int cellIdx, const myFloat blockScale, const myFloat3 p0)
{
	const myFloat3 local = d_localPos[cellIdx];
	return { p0.x + blockScale * (blockX * BLOCK_WIDTH_X + local.x),
			 p0.y + blockScale * (blockY * BLOCK_WIDTH_Y + local.y),
			 p0.z + blockScale * (blockZ * BLOCK_WIDTH_Z + local.z) };
};

__global__ void interpolate(PitchedPtr nextStep, PitchedPtr prevStep, const int4* __restrict__ laplace, const myFloat* __restrict__ hodges, const uint3 dimensions, const myFloat3 prev_p0, const myFloat3 new_p0, const myFloat prevScale, const myFloat newScale)
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
	const Complex5Vec prev = ((BlockPsis*)prevPsi)->values[dualNodeId];
	ldsPrevPsis[threadIdxInBlock].values[dualNodeId] = prev;

	// Kill also the leftover edge threads
	if (dataXid == dimensions.x || yid == dimensions.y || zid == dimensions.z)
	{
		return;
	}
	__syncthreads();

	uint primaryFace = dualNodeId * FACE_COUNT;

	myFloat3 prevPositions[FACE_COUNT + 1];
	Complex5Vec prevPsis[FACE_COUNT + 1];

	// Add the Laplacian to the Hamiltonian
#pragma unroll
	for (int i = 0; i < FACE_COUNT; ++i)
	{
		const int4 laplacian = laplace[primaryFace];

		int neighbourGlobalX = dataXid + laplacian.x;
		int neighbourGlobalY = yid + laplacian.y;
		int neighbourGlobalZ = zid + laplacian.z;

		prevPositions[i] = getGlobalPos(neighbourGlobalX, neighbourGlobalY, neighbourGlobalZ, laplacian.w, prevScale, prev_p0);

		const int neighbourThreadX = localDataXid + laplacian.x;
		const int neighbourThreadY = threadIdx.y + laplacian.y;
		const int neighbourThreadZ = threadIdx.z + laplacian.z;

		Complex5Vec otherBoundaryZeroCell;
		// Read from the local shared memory
		if ((0 <= neighbourThreadX) && (neighbourThreadX < THREAD_BLOCK_X) &&
			(0 <= neighbourThreadY) && (neighbourThreadY < THREAD_BLOCK_Y) &&
			(0 <= neighbourThreadZ) && (neighbourThreadZ < THREAD_BLOCK_Z))
		{
			const int neighbourIdx = neighbourThreadZ * THREAD_BLOCK_Y * THREAD_BLOCK_X + neighbourThreadY * THREAD_BLOCK_X + neighbourThreadX;
			otherBoundaryZeroCell = ldsPrevPsis[neighbourIdx].values[laplacian.w];
		}
		else // Read from the global memory
		{
			const int offset = laplacian.z * prevStep.slicePitch + laplacian.y * prevStep.pitch + laplacian.x * sizeof(BlockPsis);
			otherBoundaryZeroCell = ((BlockPsis*)(prevPsi + offset))->values[laplacian.w];
		}

		prevPsis[i] = otherBoundaryZeroCell;

		primaryFace++;
	}

	myFloat3 prevPos = getGlobalPos(dataXid, yid, zid, dualNodeId, prevScale, prev_p0);
	prevPositions[FACE_COUNT] = prevPos;
	prevPsis[FACE_COUNT] = prev;

	myFloat3 newPos = getGlobalPos(dataXid, yid, zid, dualNodeId, newScale, new_p0);

	myFloat maxDist = 0;
	int argMax = 0;
	for (int i = 0; i < FACE_COUNT + 1; ++i)
	{
		myFloat dist = mag(newPos - prevPositions[i]);
		if (dist > maxDist)
		{
			maxDist = dist;
			argMax = i;
		}
	}

	int closestIndices[FACE_COUNT];
	int count = 0;
	for (int i = 0; i < FACE_COUNT + 1; ++i)
	{
		if (i != argMax)
		{
			closestIndices[count] = i;
			count++;
		}
	}

	myFloat4 weights = baryCoords(prevPositions[closestIndices[0]], 
								 prevPositions[closestIndices[1]],
								 prevPositions[closestIndices[2]],
								 prevPositions[closestIndices[3]],
								 newPos);

	Complex5Vec interpolated;
	for (int i = 0; i < FACE_COUNT; ++i)
	{
		auto neighbour = prevPsis[closestIndices[i]];
		auto weight = subscript(weights, i);
		interpolated.s2 += weight * neighbour.s2;
		interpolated.s1 += weight * neighbour.s1;
		interpolated.s0 += weight * neighbour.s0;
		interpolated.s_1 += weight * neighbour.s_1;
		interpolated.s_2 += weight * neighbour.s_2;
	}

	nextPsi->values[dualNodeId] = interpolated;
}
//void energy_h(dim3 dimGrid, dim3 dimBlock, myFloat* energyPtr, PitchedPtr psi, PitchedPtr potentials, int4* lapInd, myFloat* hodges, myFloat g, uint3 dimensions, myFloat volume, size_t bodies)
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
			integrateVec << <dim3(std::ceil(newStride / 32.0), 1, 1), dim3(32, 1, 1) >> > (localAvgSpinPtr, newStride, ((newStride * 2) != prevStride), volume);
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

myFloat3 compute_p0(const myFloat block_scale, const uint xsize, const uint ysize, const uint zsize)
{
	const myFloat domainSize = block_scale * BLOCK_WIDTH_X * REPLICABLE_STRUCTURE_COUNT_X;
	const auto minp = Vector3(-domainSize * 0.5, -domainSize * 0.5, -domainSize * 0.5);
	const auto maxp = Vector3(domainSize * 0.5, domainSize * 0.5, domainSize * 0.5);

	const Vector3 domain = maxp - minp;
	const Vector3 p0 = 0.5 * (minp + maxp - block_scale * Vector3(BLOCK_WIDTH.x * xsize, BLOCK_WIDTH.y * ysize, BLOCK_WIDTH.z * zsize));
	return { p0.x, p0.y, p0.z };
}

uint integrateInTime(const myFloat block_scale, const Vector3& minp, const Vector3& maxp)
{
	// find dimensions
	const Vector3 domain = maxp - minp;
	const uint xsize = uint(domain.x / (block_scale * BLOCK_WIDTH.x)); // + 1;
	const uint ysize = uint(domain.y / (block_scale * BLOCK_WIDTH.y)); // + 1;
	const uint zsize = uint(domain.z / (block_scale * BLOCK_WIDTH.z)); // + 1;
	const Vector3 original_p0 = 0.5 * (minp + maxp - block_scale * Vector3(BLOCK_WIDTH.x * xsize, BLOCK_WIDTH.y * ysize, BLOCK_WIDTH.z * zsize));
	const myFloat3 d_original_p0 = compute_p0(block_scale, xsize, ysize, zsize);

	// compute discrete dimensions
	const uint bsize = VALUES_IN_BLOCK; // bpos.size(); // number of values inside a block

	//std::cout << "Dual 0-cells in a replicable structure: " << bsize << std::endl;
	//std::cout << "Replicable structure instances in x: " << xsize << ", y: " << ysize << ", z: " << zsize << std::endl;
	uint64_t bodies = xsize * ysize * zsize * bsize;
	//std::cout << "Dual 0-cells in total: " << bodies << std::endl;

	// Initialize device memory
	const size_t dxsize = xsize + 2; // One element buffer to both ends
	const size_t dysize = ysize + 2; // One element buffer to both ends
	const size_t dzsize = zsize + 2; // One element buffer to both ends
	cudaExtent psiExtent = make_cudaExtent(dxsize * sizeof(BlockPsis), dysize, dzsize);

	static constexpr uint32_t BUFFER_COUNT = 2;
	cudaPitchedPtr d_cudaEvenPsis[BUFFER_COUNT];
	cudaPitchedPtr d_cudaOddPsis[BUFFER_COUNT];
	for (int i = 0; i < 2; ++i)
	{
		checkCudaErrors(cudaMalloc3D(&d_cudaEvenPsis[i], psiExtent));
		checkCudaErrors(cudaMalloc3D(&d_cudaOddPsis[i], psiExtent));
	}
	cudaPitchedPtr d_cudaHPsi;
	checkCudaErrors(cudaMalloc3D(&d_cudaHPsi, psiExtent));

	myFloat* d_energy;
	myFloat* d_spinNorm;
	myFloat* d_density;
	myFloat3* d_localAvgSpin;
	myFloat3* d_u;
	myFloat3* d_v;
	myFloat* d_theta;
	checkCudaErrors(cudaMalloc(&d_energy, bodies * sizeof(myFloat)));
	checkCudaErrors(cudaMalloc(&d_spinNorm, bodies * sizeof(myFloat)));
	checkCudaErrors(cudaMalloc(&d_density, bodies * sizeof(myFloat)));
	checkCudaErrors(cudaMalloc(&d_localAvgSpin, bodies * sizeof(myFloat3)));
	checkCudaErrors(cudaMalloc(&d_u, bodies * sizeof(myFloat3)));
	checkCudaErrors(cudaMalloc(&d_v, bodies * sizeof(myFloat3)));
	checkCudaErrors(cudaMalloc(&d_theta, bodies * sizeof(myFloat)));
	PitchedPtr d_evenPsis[BUFFER_COUNT];
	PitchedPtr d_oddPsis[BUFFER_COUNT];
	for (int i = 0; i < 2; ++i)
	{
		size_t offset = d_cudaEvenPsis[i].pitch * dysize + d_cudaEvenPsis[i].pitch + sizeof(BlockPsis);
		d_evenPsis[i] = { (char*)d_cudaEvenPsis[i].ptr + offset, d_cudaEvenPsis[i].pitch, d_cudaEvenPsis[i].pitch * dysize };
		d_oddPsis[i]= { (char*)d_cudaOddPsis[i].ptr + offset, d_cudaOddPsis[i].pitch, d_cudaOddPsis[i].pitch * dysize };
	}
	size_t offset = d_cudaHPsi.pitch * dysize + d_cudaHPsi.pitch + sizeof(BlockPsis);
	PitchedPtr d_HPsi = { (char*)d_cudaHPsi.ptr + offset, d_cudaHPsi.pitch, d_cudaHPsi.pitch * dysize };

	// find terms for laplacian
	Buffer<int4> lapind;
	Buffer<myFloat> hodges;
	getLaplacian(lapind, hodges, sizeof(BlockPsis), d_evenPsis[0].pitch, d_evenPsis[0].slicePitch);

	//std::cout << "lapsize = " << lapsize << ", lapfac = " << lapfac << ", lapfac0 = " << lapfac0 << std::endl;

#if USE_QUADRUPOLE_OFFSET
	std::cout << "Quadrupole field offset is in use." << std::endl;
#else
	std::cout << "Not using quadrupole field offset." << std::endl;
#endif

	for (int i = 0; i < hodges.size(); ++i) hodges[i] = -0.5 * hodges[i]; // / (block_scale * block_scale);

	int4* d_lapind;
	checkCudaErrors(cudaMalloc(&d_lapind, lapind.size() * sizeof(int4)));

	myFloat* d_hodges;
	checkCudaErrors(cudaMalloc(&d_hodges, hodges.size() * sizeof(myFloat)));

	// Initialize host memory
	size_t hostSize = dxsize * dysize * dzsize;
	BlockPsis* h_evenPsi;
	BlockPsis* h_oddPsi;
	checkCudaErrors(cudaMallocHost(&h_evenPsi, hostSize * sizeof(BlockPsis)));
	checkCudaErrors(cudaMallocHost(&h_oddPsi, hostSize * sizeof(BlockPsis)));
	memset(h_evenPsi, 0, hostSize * sizeof(BlockPsis));
	memset(h_oddPsi, 0, hostSize * sizeof(BlockPsis));

	myFloat* h_density;
	myFloat3* h_u;
	myFloat* h_theta;
	myFloat3* h_localAvgSpin;
	checkCudaErrors(cudaMallocHost(&h_density, bodies * sizeof(myFloat)));
	checkCudaErrors(cudaMallocHost(&h_u, bodies * sizeof(myFloat3)));
	checkCudaErrors(cudaMallocHost(&h_theta, bodies * sizeof(myFloat)));
	checkCudaErrors(cudaMallocHost(&h_localAvgSpin, bodies * sizeof(myFloat3)));

#if COMPUTE_GROUND_STATE
	// Initialize discrete field
	std::ifstream fs(GROUND_STATE_FILENAME, std::ios::binary | std::ios::in);
	if (fs.fail() != 0)
	{
		std::cout << "Initialized ground state with random noise." << std::endl;

		std::default_random_engine generator(std::chrono::high_resolution_clock::now().time_since_epoch().count());
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
						const myFloat2 s2{ distribution(generator), distribution(generator) };
						const myFloat2 s1{ distribution(generator), distribution(generator) };
						const myFloat2 s0{ distribution(generator), distribution(generator) };
						const myFloat2 s_1{ distribution(generator), distribution(generator) };
						const myFloat2 s_2{ distribution(generator), distribution(generator) };
						h_evenPsi[dstI].values[l].s2 = s2;
						h_evenPsi[dstI].values[l].s1 = s1;
						h_evenPsi[dstI].values[l].s0 = s0;
						h_evenPsi[dstI].values[l].s_1 = s_1;
						h_evenPsi[dstI].values[l].s_2 = s_2;
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
		std::normal_distribution<myFloat> distribution(0.0, 1.0);

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
						const myFloat2 rand = { distribution(generator), distribution(generator) };
						const myFloat dens = (conj(h_oddPsi[dstI].values[l].s0) * h_oddPsi[dstI].values[l].s0).x;
						h_oddPsi[dstI].values[l].s0 += sqrt(dens) * NOISE_AMPLITUDE * rand;

						// Normalize
						const myFloat newDens = (conj(h_oddPsi[dstI].values[l].s0) * h_oddPsi[dstI].values[l].s0).x;
						h_oddPsi[dstI].values[l].s0 = sqrt(dens / newDens) * h_oddPsi[dstI].values[l].s0;
					}
				}
			}
		}
	}
	std::cout << "Initial noise applied." << std::endl;
#else
	std::cout << "No initial noise." << std::endl;
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
	h_cudaEvenPsi.xsize = d_cudaEvenPsis[0].xsize;
	h_cudaEvenPsi.ysize = d_cudaEvenPsis[0].ysize;

	h_cudaOddPsi.ptr = h_oddPsi;
	h_cudaOddPsi.pitch = dxsize * sizeof(BlockPsis);
	h_cudaOddPsi.xsize = d_cudaOddPsis[0].xsize;
	h_cudaOddPsi.ysize = d_cudaOddPsis[0].ysize;

	// Copy from host memory to device memory
	cudaMemcpy3DParms evenPsiParams = { 0 };
	cudaMemcpy3DParms oddPsiParams = { 0 };

	evenPsiParams.srcPtr = h_cudaEvenPsi;
	evenPsiParams.dstPtr = d_cudaEvenPsis[0];
	evenPsiParams.extent = psiExtent;
	evenPsiParams.kind = cudaMemcpyHostToDevice;

	oddPsiParams.srcPtr = h_cudaOddPsi;
	oddPsiParams.dstPtr = d_cudaOddPsis[0];
	oddPsiParams.extent = psiExtent;
	oddPsiParams.kind = cudaMemcpyHostToDevice;

	checkCudaErrors(cudaMemcpy3D(&evenPsiParams));
	checkCudaErrors(cudaMemcpy3D(&oddPsiParams));
	checkCudaErrors(cudaMemcpy(d_lapind, &lapind[0], lapind.size() * sizeof(int4), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_hodges, &hodges[0], hodges.size() * sizeof(myFloat), cudaMemcpyHostToDevice));

	// Clear host memory after data has been copied to devices
	cudaDeviceSynchronize();
	lapind.clear();
	hodges.clear();
#if !(SAVE_PICTURE)
	cudaFreeHost(h_evenPsi);
	cudaFreeHost(h_oddPsi);
#endif
	cudaMemcpy3DParms evenPsiBackParams = { 0 };
	evenPsiBackParams.srcPtr = d_cudaEvenPsis[0];
	evenPsiBackParams.dstPtr = h_cudaEvenPsi;
	evenPsiBackParams.extent = psiExtent;
	evenPsiBackParams.kind = cudaMemcpyDeviceToHost;

	cudaMemcpy3DParms oddPsiBackParams = { 0 };
	oddPsiBackParams.srcPtr = d_cudaOddPsis[0];
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
	MagFields Bs{ 0 };

	myFloat volume = block_scale * block_scale * block_scale * VOLUME;

	if (loadGroundState)
	{
		switch (initPhase)
		{
		case Phase::UN:
			std::cout << "Transform ground state to uniaxial nematic phase." << std::endl;
			unState << <dimGrid, dimBlock >> > (d_oddPsis[0], dimensions);
			break;
		case Phase::BN_VERT:
			std::cout << "Transform ground state to vertically oriented biaxial nematic phase with a phase of " << relativePhase << "." << std::endl;
			verticalBnState << <dimGrid, dimBlock >> > (d_oddPsis[0], dimensions, relativePhase);
			break;
		case Phase::BN_HORI:
			std::cout << "Transform ground state to horizontally oriented biaxial nematic phase with a phase of " << relativePhase << "." << std::endl;
			horizontalBnState << <dimGrid, dimBlock >> > (d_oddPsis[0], dimensions, relativePhase);
			break;
		case Phase::CYCLIC:
			std::cout << "Transform ground state to cyclic phase with a phase of " << relativePhase << "." << std::endl;
			cyclicState << <dimGrid, dimBlock >> > (d_oddPsis[0], dimensions, relativePhase);
			break;
		default:
			std::cout << "Initial phase " << (int)initPhase << " is not supported!";
			break;
		}

		std::cout << "Total density: " << getDensity(dimGrid, dimBlock, d_density, d_oddPsis[0], dimensions, bodies, volume) << std::endl;
	}

	// Take one forward Euler step if starting from the ground state or time step changed
	if (doForward)
	{
		std::cout << "No even time step file found. Doing one forward step." << std::endl;

		signal = getSignal(t);
		Bs.Bq = BqScale * signal.Bq;
		Bs.Bb = BzScale * signal.Bb;
		Bs.BqQuad = BqQuadScale * signal.Bq;
		Bs.BbQuad = BzQuadScale * signal.Bb;
		forwardEuler << <dimGrid, dimBlock >> > (d_evenPsis[0], d_oddPsis[0], d_lapind, d_hodges, Bs, dimensions, block_scale, d_original_p0, c0, c2, c4, alpha, t);
	}
	else
	{
		std::cout << "Skipping the forward step." << std::endl;
	}

#if COMPUTE_GROUND_STATE
	uint iter = 0;

	normalize_h(dimGrid, dimBlock, d_density, d_evenPsis[0], dimensions, bodies, volume);

	while (true)
	{
		if ((iter % 10000) == 0) std::cout << "Iteration " << iter << std::endl;
#if SAVE_PICTURE
		if ((iter % 100) == 0)
		{
			checkCudaErrors(cudaMemcpy3D(&evenPsiBackParams));
			signal = getSignal(0);
			Bs.Bq = BqScale * signal.Bq;
			Bs.Bb = BzScale * signal.Bb;
			drawIandR("GS", h_evenPsi, dxsize, dysize, dzsize, iter, Bs, d_original_p0, block_scale);
			std::cout << "Normalized particle count: " << getDensity(dimGrid, dimBlock, d_density, d_evenPsis[0], dimensions, bodies, volume) << std::endl;

			myFloat3 com = centerOfMass(h_evenPsi, bsize, dxsize, dysize, dzsize, block_scale, d_original_p0);
			std::cout << "Center of mass: " << com.x << ", " << com.y << ", " << com.z << std::endl;

#if 1
			// Compute energy/chemical potential // Alice ring experimen E = 127.295
			innerProduct << <dimGrid, dimBlock >> > (d_energy, d_evenPsis[0], d_HPsi, dimensions);
			int prevStride = bodies;
			while (prevStride > 1)
			{
				int newStride = prevStride / 2;
				integrate << <dim3(std::ceil(newStride / 32.0), 1, 1), dim3(32, 1, 1) >> > (d_energy, newStride, ((newStride * 2) != prevStride), volume);
				prevStride = newStride;
			}
			myFloat hEnergy = 0;
			checkCudaErrors(cudaMemcpy(&hEnergy, d_energy, sizeof(myFloat), cudaMemcpyDeviceToHost));
			std::cout << "Energy: " << hEnergy << std::endl;
#endif
		}
#endif
		if (iter == GROUND_STATE_ITERATION_COUNT)
		{
			checkCudaErrors(cudaMemcpy3D(&evenPsiBackParams));
			std::ofstream fs(GROUND_STATE_FILENAME, std::ios::binary | std::ios_base::trunc);
			if (fs.fail() != 0) return 1;
			fs.write((char*)&h_evenPsi[0], hostSize * sizeof(BlockPsis));
			fs.close();
			return 0;
		}
		// Take an imaginary time step
		itp << <dimGrid, dimBlock >> > (d_HPsi, d_oddPsis[0], d_evenPsis[0], d_lapind, d_hodges, {0}, dimensions, block_scale, d_original_p0, c0, c2, c4, t);
		// Normalize
		normalize_h(dimGrid, dimBlock, d_density, d_oddPsis[0], dimensions, bodies, volume);

		// Take an imaginary time step
		itp << <dimGrid, dimBlock >> > (d_HPsi, d_evenPsis[0], d_oddPsis[0], d_lapind, d_hodges, { 0 }, dimensions, block_scale, d_original_p0, c0, c2, c4, t);
		// Normalize
		normalize_h(dimGrid, dimBlock, d_density, d_evenPsis[0], dimensions, bodies, volume);

		//energy_h(dimGrid, dimBlock, d_energy, d_evenPsi, d_pot, d_lapind, d_hodges, g, dimensions, volume, bodies);
		//myFloat hDensity = 0;
		//myFloat hEnergy = 0;
		//checkCudaErrors(cudaMemcpy(&hDensity, d_density, sizeof(myFloat), cudaMemcpyDeviceToHost));
		//checkCudaErrors(cudaMemcpy(&hEnergy, d_energy, sizeof(myFloat), cudaMemcpyDeviceToHost));

		//myFloat newMu = hEnergy / hDensity;
		//myFloat newE = hEnergy;
		//
		//std::cout << "Total density: " << hDensity << ", Total energy: " << hEnergy << ", mu: " << newMu << std::endl;

		//if (std::abs(mu - newMu) < 1e-4) break;

		//mu = newMu;
		//E = newE;

		iter++;
	}

#else
	std::string tString = std::string("t = [");
	std::string bqString = std::string("Bq = [");
	std::string bbString = std::string("Bz = [");
	std::string optTrapString = std::string("opt_trap = [");
	std::string spinString = std::string("Spin = [Spin");
	std::string magX = std::string("mag_x = [mag_x");
	std::string magY = std::string("mag_y = [mag_y");
	std::string magZ = std::string("mag_z = [mag_z");
	std::string densityStr = std::string("dens = [");

	int lastSaveTime = 0;

#if _WIN32
	std::string dirSeparator = "\\";
	std::string mkdirOptions = "";
#else
	std::string dirSeparator = "/";
	std::string mkdirOptions = "-p ";
#endif

	std::string dirPrefix = phaseToString(initPhase) + dirSeparator + toStringShort(HOLD_TIME)+"us_winding" + dirSeparator + toString(relativePhase / PI * 180.0, 2) + "_deg_phase" + dirSeparator + getProjectionString() + dirSeparator;

	std::string densDir = dirPrefix; // +"dens";
	//std::string vtksDir = dirPrefix + "dens_vtks";
	//std::string spinorVtksDir = dirPrefix + "spinor_vtks";
	std::string datsDir = dirPrefix + "dats";
	//
	std::string createResultsDirCommand = "mkdir " + mkdirOptions + densDir;
	//std::string createVtksDirCommand = "mkdir " + mkdirOptions + vtksDir;
	//std::string createSpinorVtksDirCommand = "mkdir " + mkdirOptions + spinorVtksDir;
	std::string createDatsDirCommand = "mkdir " + mkdirOptions + datsDir;
	system(createResultsDirCommand.c_str());
	//system(createVtksDirCommand.c_str());
	//system(createSpinorVtksDirCommand.c_str());
	//system(createDatsDirCommand.c_str());

	myFloat expansionBlockScale = block_scale;
	myFloat3 expansion_p0 = d_original_p0;

	// Measure wall clock time
	static auto prevTime = std::chrono::high_resolution_clock::now();

	while (t < STATE_PREP_DURATION)
	{
		// update odd values
		t += dt / omega_r * 1e3; // [ms]
		signal = getSignal(t);
		Bs.Bq = BqScale * signal.Bq;
		Bs.Bb = BzScale * signal.Bb;
		Bs.BqQuad = BqQuadScale * signal.Bq;
		Bs.BbQuad = BzQuadScale * signal.Bb;
		leapfrog << <dimGrid, dimBlock >> > (d_oddPsis[0], d_evenPsis[0], d_lapind, d_hodges, Bs, dimensions, expansionBlockScale, d_original_p0, c0, c2, c4, alpha, t);
		//densityStr += std::to_string(getDensity(dimGrid, dimBlock, d_density, d_oddPsi, dimensions, bodies, volume)) + ", ";
		//tString += std::to_string(t) + ", ";
		//std::cout << std::to_string(signal.Bq) + ", ";
		//bbString += std::to_string(signal.Bb.z) + ", ";
		//optTrapString += std::to_string(trap({ maxp.x, maxp.y, maxp.z }, t)) + ", ";

		// update even values
		t += dt / omega_r * 1e3; // [ms]
		signal = getSignal(t);
		Bs.Bq = BqScale * signal.Bq;
		Bs.Bb = BzScale * signal.Bb;
		Bs.BqQuad = BqQuadScale * signal.Bq;
		Bs.BbQuad = BzQuadScale * signal.Bb;
		leapfrog << <dimGrid, dimBlock >> > (d_evenPsis[0], d_oddPsis[0], d_lapind, d_hodges, Bs, dimensions, expansionBlockScale, d_original_p0, c0, c2, c4, alpha, t);
		//densityStr += std::to_string(getDensity(dimGrid, dimBlock, d_density, d_evenPsi, dimensions, bodies, volume)) + ", ";
		//tString += std::to_string(t) + ", ";
		//std::cout << std::to_string(signal.Bq) + ", ";
		//bbString += std::to_string(signal.Bb.z) + ", ";
		//optTrapString += std::to_string(trap({ maxp.x, maxp.y, maxp.z }, t)) + ", ";
	}

#if SAVE_PICTURE
	// Copy back from device memory to host memory
	checkCudaErrors(cudaMemcpy3D(&oddPsiBackParams));

	// Measure wall clock time
	auto duration = std::chrono::high_resolution_clock::now() - prevTime;
	std::cout << "Simulation time: " << t << " ms. Real time from previous save: " << duration.count() * 1e-9 << " s." << std::endl;
	prevTime = std::chrono::high_resolution_clock::now();

	signal = getSignal(0);
	Bs.Bq = BqScale * signal.Bq;
	Bs.Bb = BzScale * signal.Bb;
	drawDensity(densDir, h_oddPsi, dxsize, dysize, dzsize, t - STATE_PREP_DURATION, Bs, d_original_p0, expansionBlockScale);
#endif

	uint32_t bufferIdx = 0;

	while (t < END_TIME)
	{
		// integrate one iteration
		for (uint step = 0; step < IMAGE_SAVE_FREQUENCY; step++)
		{
			// update odd values
			t += dt / omega_r * 1e3; // [ms]
			if (t >= GRID_SCALING_START)
			{
				const myFloat prevScale = expansionBlockScale;
				const myFloat3 prev_p0 = expansion_p0;

				expansionBlockScale += dt / omega_r * 1e3 * k * block_scale;
				expansion_p0 = compute_p0(expansionBlockScale, xsize, ysize, zsize);
				volume = expansionBlockScale * expansionBlockScale * block_scale * VOLUME;

				const uint32_t nextBufferIdx = (bufferIdx + 1) % BUFFER_COUNT;

				interpolate << <dimGrid, dimBlock >> > (d_evenPsis[nextBufferIdx], d_evenPsis[bufferIdx], d_lapind, d_hodges, dimensions, prev_p0, expansion_p0, prevScale, expansionBlockScale);
				normalize_h(dimGrid, dimBlock, d_density, d_evenPsis[nextBufferIdx], dimensions, bodies, volume);
				interpolate << <dimGrid, dimBlock >> > (d_oddPsis[nextBufferIdx], d_oddPsis[bufferIdx], d_lapind, d_hodges, dimensions, prev_p0, expansion_p0, prevScale, expansionBlockScale);
				normalize_h(dimGrid, dimBlock, d_density, d_oddPsis[nextBufferIdx], dimensions, bodies, volume);

				bufferIdx = nextBufferIdx;
			}
			signal = getSignal(t);
			Bs.Bq = BqScale * signal.Bq;
			Bs.Bb = BzScale * signal.Bb;
			Bs.BqQuad = BqQuadScale * signal.Bq;
			Bs.BbQuad = BzQuadScale * signal.Bb;
			leapfrog << <dimGrid, dimBlock >> > (d_oddPsis[bufferIdx], d_evenPsis[bufferIdx], d_lapind, d_hodges, Bs, dimensions, expansionBlockScale, expansion_p0, c0, c2, c4, alpha, t);

			// update even values
			t += dt / omega_r * 1e3; // [ms]
			if (t >= GRID_SCALING_START)
			{
				const myFloat prevScale = expansionBlockScale;
				const myFloat3 prev_p0 = expansion_p0;

				expansionBlockScale += dt / omega_r * 1e3 * k * block_scale;
				expansion_p0 = compute_p0(expansionBlockScale, xsize, ysize, zsize);
				volume = expansionBlockScale * expansionBlockScale * block_scale * VOLUME;

				const uint32_t nextBufferIdx = (bufferIdx + 1) % BUFFER_COUNT;

				interpolate << <dimGrid, dimBlock >> > (d_evenPsis[nextBufferIdx], d_evenPsis[bufferIdx], d_lapind, d_hodges, dimensions, prev_p0, expansion_p0, prevScale, expansionBlockScale);
				normalize_h(dimGrid, dimBlock, d_density, d_evenPsis[nextBufferIdx], dimensions, bodies, volume);
				interpolate << <dimGrid, dimBlock >> > (d_oddPsis[nextBufferIdx], d_oddPsis[bufferIdx], d_lapind, d_hodges, dimensions, prev_p0, expansion_p0, prevScale, expansionBlockScale);
				normalize_h(dimGrid, dimBlock, d_density, d_oddPsis[nextBufferIdx], dimensions, bodies, volume);

				bufferIdx = nextBufferIdx;
			}
			signal = getSignal(t);
			Bs.Bq = BqScale * signal.Bq;
			Bs.Bb = BzScale * signal.Bb;
			Bs.BqQuad = BqQuadScale * signal.Bq;
			Bs.BbQuad = BzQuadScale * signal.Bb;
			leapfrog << <dimGrid, dimBlock >> > (d_evenPsis[bufferIdx], d_oddPsis[bufferIdx], d_lapind, d_hodges, Bs, dimensions, expansionBlockScale, expansion_p0, c0, c2, c4, alpha, t);
		}
#if SAVE_PICTURE
		std::cout << "N = " << getDensity(dimGrid, dimBlock, d_density, d_oddPsis[0], dimensions, bodies, volume) << std::endl;

		// Copy back from device memory to host memory
		checkCudaErrors(cudaMemcpy3D(&oddPsiBackParams));

		// Measure wall clock time
		auto duration = std::chrono::high_resolution_clock::now() - prevTime;
		std::cout << "Simulation time: " << t << " ms. Real time from previous save: " << duration.count() * 1e-9 << " s." << std::endl;
		prevTime = std::chrono::high_resolution_clock::now();

		signal = getSignal(0);
		Bs.Bq = BqScale * signal.Bq;
		Bs.Bb = BzScale * signal.Bb;
		drawDensity(densDir, h_oddPsi, dxsize, dysize, dzsize, t - STATE_PREP_DURATION, Bs, expansion_p0, expansionBlockScale);
#endif
#if SAVE_STATES
		// Copy back from device memory to host memory
		//checkCudaErrors(cudaMemcpy3D(&oddPsiBackParams));

		//if (t - STATE_PREP_DURATION >= 179)
		static bool savedState = false;
		if (t >= 15.0 && !savedState)
		{
			//saveVolume(vtksDir, h_oddPsi, bsize, dxsize, dysize, dzsize, expansionBlockScale, d_p0, t - STATE_PREP_DURATION);
			//saveSpinor(spinorVtksDir, h_oddPsi, bsize, dxsize, dysize, dzsize, expansionBlockScale, d_p0, t - STATE_PREP_DURATION);

			std::ofstream oddFs(datsDir + "/" + toString(t) + ".dat", std::ios::binary | std::ios_base::trunc);
			if (oddFs.fail() != 0) return 1;
			oddFs.write((char*)&h_oddPsi[0], hostSize * sizeof(BlockPsis));
			oddFs.close();

			checkCudaErrors(cudaMemcpy3D(&evenPsiBackParams));
			std::ofstream evenFs(datsDir + "/" + "even_" + toString(t) + ".dat", std::ios::binary | std::ios_base::trunc);
			if (evenFs.fail() != 0) return 1;
			evenFs.write((char*)&h_evenPsi[0], hostSize * sizeof(BlockPsis));
			evenFs.close();

			savedState = true;
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

	for (int i = 0; i < BUFFER_COUNT; ++i)
	{
		checkCudaErrors(cudaFree(d_cudaEvenPsis[i].ptr));
		checkCudaErrors(cudaFree(d_cudaOddPsis[i].ptr));
	}
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

void readConfFile(const std::string& confFileName)
{
	std::cout << "Read conf file " << confFileName << std::endl;
	std::ifstream file;
	file.open(confFileName, std::ios::in);
	if (file.is_open())
	{
		std::string line;
		while (std::getline(file, line))
		{
			if (size_t pos = line.find("rel_phase") != std::string::npos)
			{
				std::cout << "Relative phase from conf file: " << line.substr(pos + 9) << std::endl;
				relativePhase = std::stod(line.substr(pos + 9));
			}
			else if (size_t pos = line.find("phase") != std::string::npos)
			{
				std::cout << "Phase from conf file: " << line.substr(pos + 5) << std::endl;
				initPhase = stringToPhase(line.substr(pos + 5));
			}
		}
	}
}

int main(int argc, char** argv)
{
	constexpr myFloat blockScale = DOMAIN_SIZE_X / REPLICABLE_STRUCTURE_COUNT_X / BLOCK_WIDTH_X;

	if (argc > 1)
	{
		std::cout << "Read config " << argv[1] << std::endl;
		readConfFile(std::string(argv[1]));
	}

	std::cout << "Start simulating from t = " << t << " ms, with a time step size of " << dt << "." << std::endl;
	std::cout << "The simulation will end at " << END_TIME << " ms." << std::endl;
	//std::cout << "Block scale = " << blockScale << std::endl;
	//std::cout << "Dual edge length = " << DUAL_EDGE_LENGTH * blockScale << std::endl;
	//std::cout << "c0: " << c0 << ", c2: " << c2 << ", c4: " << c4 << std::endl;
	std::cout << "Three-body loss magnitude: " << alpha << std::endl;
#if USE_QUADRATIC_ZEEMAN
	std::cout << "Taking the quadratic Zeeman term into account" << std::endl;
#else
	std::cout << "No quadratic Zeeman term" << std::endl;
#endif

	printRamp();
	printBasis();

	// integrate in time using DEC
	auto domainMin = Vector3(-DOMAIN_SIZE_X * 0.5, -DOMAIN_SIZE_Y * 0.5, -DOMAIN_SIZE_Z * 0.5);
	auto domainMax = Vector3(DOMAIN_SIZE_X * 0.5, DOMAIN_SIZE_Y * 0.5, DOMAIN_SIZE_Z * 0.5);

	//Phase phases[] = {Phase::BN_VERT, Phase::BN_HORI, Phase::CYCLIC};
	//Phase phases[] = {Phase::BN_VERT};
	//for (auto phase : phases)
	//{
	//	initPhase = phase;
	//	t = 0;
	//	integrateInTime(blockScale, domainMin, domainMax);
	//}

	//constexpr int TURNS = 8; // 45 degree global phase turns
	//for (int turn = 0; turn < TURNS; ++turn)
	//{
	//	relativePhase = turn * 45.0 / 180.0 * PI;
	//	t = 0;
	//	integrateInTime(blockScale, domainMin, domainMax);
	//}

	integrateInTime(blockScale, domainMin, domainMax);

	return 0;
}