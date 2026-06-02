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

std::string toStringShort(const myFloat value)
{
	std::ostringstream out;
	out.precision(2);
	out << std::fixed << value;
	return out.str();
};

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

#define SAVE_STATES 1
#define SAVE_PICTURE 1

#define THREAD_BLOCK_X 16
#define THREAD_BLOCK_Y 2
#define THREAD_BLOCK_Z 1

bool USE_QUADRUPOLE_OFFSET = false;
bool USE_INITIAL_NOISE = false;

bool USE_QUADRATIC_ZEEMAN = false;
bool USE_THREE_BODY_LOSS = false;

constexpr myFloat DOMAIN_SIZE_X = 20.0;
constexpr myFloat DOMAIN_SIZE_Y = 20.0;
constexpr myFloat DOMAIN_SIZE_Z = 20.0;

constexpr myFloat REPLICABLE_STRUCTURE_COUNT_X = 112.0; // 58.0 + 3 * 6.0;
//constexpr myFloat REPLICABLE_STRUCTURE_COUNT_Y = 112.0;
//constexpr myFloat REPLICABLE_STRUCTURE_COUNT_Z = 112.0;

constexpr myFloat N = 2e5; // Number of atoms in the condensate

constexpr myFloat trapFreq_r = 126;
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

constexpr myFloat myGamma = 2.9e-30;
const myFloat alpha = N * N * myGamma * 1e-12 / (a_r * a_r * a_r * a_r * a_r * a_r * 2 * PI * trapFreq_r);

constexpr myFloat muB = 9.27400968e-24; // [m^2 kg / s^2 T^-1] Bohr magneton

const myFloat BqScale = -(0.5 * muB / (hbar * omega_r) * a_r) / 100.; // [cm/Gauss]
constexpr myFloat BzScale = -(0.5 * muB / (hbar * omega_r)) / 10000.; // [1/Gauss]

constexpr myFloat A_hfs = 3.41734130545215;
const myFloat BqQuadScale = 100 * a_r * sqrt(0.25 * 1000 * (1.399624624 * 1.399624624) / (trapFreq_r * 2 * A_hfs)); //[cm/Gauss]
const myFloat BzQuadScale = sqrt(0.25 * 1000 * (1.399624624 * 1.399624624) / (trapFreq_r * 2 * A_hfs)); //[1/Gauss]  \sqrt{g_q}

constexpr myFloat SQRT_2 = 1.41421356237309;
constexpr myFloat INV_SQRT_2 = 0.70710678118655;

const std::string EXTRA_INFORMATION = toStringShort(DOMAIN_SIZE_X) + "_" + toStringShort(REPLICABLE_STRUCTURE_COUNT_X);
const std::string GROUND_STATE_FILENAME = "ground_state_psi_" + EXTRA_INFORMATION + "_" + PRECISION + ".dat";
const std::string SAVE_FILE_PREFIX = "";

constexpr myFloat NOISE_AMPLITUDE = 0.1;

myFloat dt = 5e-5; // 1 x // Before the monopole creation ramp (0 - 200 ms)
//myFloat dt = 1e-5; // 0.1 x // During and after the monopole creation ramp (200 ms - )

const myFloat IMAGE_SAVE_INTERVAL = 0.01; // ms
uint IMAGE_SAVE_FREQUENCY = uint(IMAGE_SAVE_INTERVAL * 0.5 / 1e3 * omega_r / dt) + 1;

const uint STATE_SAVE_INTERVAL = 10.0; // ms

myFloat t = 0; // Start time in ms
myFloat END_TIME = 0.6; // End time in ms

myFloat POLAR_FERRO_MIX = 0.0;

__device__ __inline__ myFloat trap(myFloat3 p, myFloat t)
{
	if (t >= EXPANSION_START) {
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

__device__ __inline__ myFloat3 magneticField(myFloat3 p, myFloat Bq, myFloat3 Bb, bool USE_QUADRUPOLE_OFFSET)
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

__global__ void maxHamilton(myFloat* maxHamlPtr, PitchedPtr prevStep, MagFields Bs, uint3 dimensions, myFloat block_scale, myFloat3 p0, myFloat c0, myFloat c2, myFloat alpha, myFloat t)
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

	const myFloat normSq_s1 = prev.s1.x * prev.s1.x + prev.s1.y * prev.s1.y;
	const myFloat normSq_s0 = prev.s0.x * prev.s0.x + prev.s0.y * prev.s0.y;
	const myFloat normSq_s_1 = prev.s_1.x * prev.s_1.x + prev.s_1.y * prev.s_1.y;
	const myFloat normSq = normSq_s1 + normSq_s0 + normSq_s_1;

	const myFloat3 localPos = d_localPos[dualNodeId];
	const myFloat3 globalPos = { p0.x + block_scale * (dataXid * BLOCK_WIDTH_X + localPos.x),
		p0.y + block_scale * (yid * BLOCK_WIDTH_Y + localPos.y),
		p0.z + block_scale * (zid * BLOCK_WIDTH_Z + localPos.z) };
	const myFloat totalPot = trap(globalPos, t) + c0 * normSq;

	myFloat3 hamilton = { totalPot, totalPot, totalPot };

	const myFloat2 temp = SQRT_2 * (conj(prev.s1) * prev.s0 + conj(prev.s0) * prev.s_1);
	const myFloat3 magnetization = { temp.x, temp.y, normSq_s1 - normSq_s_1 };
	myFloat3 B = magneticField(globalPos, Bs.Bq, Bs.Bb, false);
	B += c2 * magnetization;

	// Linear Zeeman shift
	hamilton.x += abs(INV_SQRT_2 * B.x);
	hamilton.y += abs(INV_SQRT_2 * B.y);
	hamilton.z += abs(B.z);

	size_t idx = zid * dimensions.x * dimensions.y * VALUES_IN_BLOCK + yid * dimensions.x * VALUES_IN_BLOCK + dataXid * VALUES_IN_BLOCK + dualNodeId;
	maxHamlPtr[idx] = max(hamilton.x, max(hamilton.y, hamilton.z));
};

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
	Complex3Vec psi = ((BlockPsis*)pPsi)->values[dualNodeId];

	size_t idx = VALUES_IN_BLOCK * (zid * dimensions.x * dimensions.y + yid * dimensions.x + dataXid) + dualNodeId;
	density[idx] = (psi.s1 * conj(psi.s1)).x + (psi.s0 * conj(psi.s0)).x + (psi.s_1 * conj(psi.s_1)).x;
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

	Complex3Vec left = ((BlockPsis*)(pLeft.ptr + pLeft.slicePitch * zid + pLeft.pitch * yid) + dataXid)->values[dualNodeId];
	Complex3Vec right = ((BlockPsis*)(pRight.ptr + pRight.slicePitch * zid + pRight.pitch * yid) + dataXid)->values[dualNodeId];

	size_t idx = VALUES_IN_BLOCK * (zid * dimensions.x * dimensions.y + yid * dimensions.x + dataXid) + dualNodeId;
	result[idx] = (conj(left.s1) * right.s1).x + (conj(left.s0) * right.s0).x + (conj(left.s_1) * right.s_1).x;
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
	Complex3Vec psi = ((BlockPsis*)pPsi)->values[dualNodeId];

	myFloat normSq_s1 = psi.s1.x * psi.s1.x + psi.s1.y * psi.s1.y;
	myFloat normSq_s0 = psi.s0.x * psi.s0.x + psi.s0.y * psi.s0.y;
	myFloat normSq_s_1 = psi.s_1.x * psi.s_1.x + psi.s_1.y * psi.s_1.y;

	myFloat density = normSq_s1 + normSq_s0 + normSq_s_1;

	psi.s1 = psi.s1 / sqrt(density);
	psi.s0 = psi.s0 / sqrt(density);
	psi.s_1 = psi.s_1 / sqrt(density);

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
	Complex3Vec psi = ((BlockPsis*)pPsi)->values[dualNodeId];

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
	if (theta < 0) {
		theta += PI;
	}

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
	Complex3Vec psi = blockPsis->values[dualNodeId];
	myFloat sqrtDens = sqrt(density[0]);
	psi.s1 = psi.s1 / sqrtDens;
	psi.s0 = psi.s0 / sqrtDens;
	psi.s_1 = psi.s_1 / sqrtDens;

	blockPsis->values[dualNodeId] = psi;
}

__global__ void weightedDiff(myFloat* result, PitchedPtr pLeft, PitchedPtr pRight, uint3 dimensions)
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

	Complex3Vec diff = { right.s1 - left.s1, right.s0 - left.s0, right.s_1 - left.s_1 };

	myFloat leftSqr = (conj(left.s1) * left.s1).x + (conj(left.s0) * left.s0).x + (conj(left.s_1) * left.s_1).x;
	myFloat diffSqr = (conj(diff.s1) * diff.s1).x + (conj(diff.s0) * diff.s0).x + (conj(diff.s_1) * diff.s_1).x;

	size_t idx = VALUES_IN_BLOCK * (zid * dimensions.x * dimensions.y + yid * dimensions.x + dataXid) + dualNodeId;
	result[idx] = leftSqr * diffSqr;
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

	myFloat normSq_s1 = prev.s1.x * prev.s1.x + prev.s1.y * prev.s1.y;
	myFloat normSq_s0 = prev.s0.x * prev.s0.x + prev.s0.y * prev.s0.y;
	myFloat normSq_s_1 = prev.s_1.x * prev.s_1.x + prev.s_1.y * prev.s_1.y;
	myFloat normSq = normSq_s1 + normSq_s0 + normSq_s_1;

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

	myFloat normSq_s1 = prev.s1.x * prev.s1.x + prev.s1.y * prev.s1.y;
	myFloat normSq_s0 = prev.s0.x * prev.s0.x + prev.s0.y * prev.s0.y;
	myFloat normSq_s_1 = prev.s_1.x * prev.s_1.x + prev.s_1.y * prev.s_1.y;
	myFloat normSq = normSq_s1 + normSq_s0 + normSq_s_1;

	pPsi->values[dualNodeId].s1 = { sqrt(normSq), 0 };
	pPsi->values[dualNodeId].s0 = { 0, 0 };
	pPsi->values[dualNodeId].s_1 = { 0, 0 };
};

__global__ void mixedState(PitchedPtr psi, const uint3 dimensions, const myFloat polarFerroMix)
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

	myFloat normSq_s1 = prev.s1.x * prev.s1.x + prev.s1.y * prev.s1.y;
	myFloat normSq_s0 = prev.s0.x * prev.s0.x + prev.s0.y * prev.s0.y;
	myFloat normSq_s_1 = prev.s_1.x * prev.s_1.x + prev.s_1.y * prev.s_1.y;
	myFloat normSq = normSq_s1 + normSq_s0 + normSq_s_1;

	pPsi->values[dualNodeId].s1 = { sqrt(normSq * polarFerroMix), 0 };
	pPsi->values[dualNodeId].s0 = { sqrt(normSq * (1.0f - polarFerroMix)), 0 };
	pPsi->values[dualNodeId].s_1 = { 0, 0 };
};

#if COMPUTE_GROUND_STATE
__global__ void itp(PitchedPtr HPsiPtr, PitchedPtr nextStep, PitchedPtr prevStep, const int4* __restrict__ laplace, const myFloat* __restrict__ hodges, MagFields Bs, const uint3 dimensions, const myFloat block_scale, const myFloat3 p0, const myFloat c0, const myFloat c2, const myFloat dt, const myFloat t)
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

		const myFloat hodge = hodges[primaryFace] / (block_scale * block_scale);
		H.s1 += hodge * (otherBoundaryZeroCell.s1 - prev.s1);
		H.s0 += hodge * (otherBoundaryZeroCell.s0 - prev.s0);
		H.s_1 += hodge * (otherBoundaryZeroCell.s_1 - prev.s_1);

		primaryFace++;
	}

	const myFloat normSq_s1 = prev.s1.x * prev.s1.x + prev.s1.y * prev.s1.y;
	const myFloat normSq_s_1 = prev.s_1.x * prev.s_1.x + prev.s_1.y * prev.s_1.y;
	const myFloat normSq = normSq_s1 + (prev.s0.x * prev.s0.x + prev.s0.y * prev.s0.y) + normSq_s_1;

	const myFloat3 localPos = d_localPos[dualNodeId];
	const myFloat3 globalPos = { p0.x + block_scale * (dataXid * BLOCK_WIDTH_X + localPos.x),
		p0.y + block_scale * (yid * BLOCK_WIDTH_Y + localPos.y),
		p0.z + block_scale * (zid * BLOCK_WIDTH_Z + localPos.z) };
	const myFloat totalPot = trap(globalPos, t) + c0 * normSq;

	H.s1 += totalPot * prev.s1;
	H.s0 += totalPot * prev.s0;
	H.s_1 += totalPot * prev.s_1;

	const myFloat2 magXY = SQRT_2 * (conj(prev.s1) * prev.s0 + conj(prev.s0) * prev.s_1);
	myFloat3 B = c2 * myFloat3{ magXY.x, magXY.y, normSq_s1 - normSq_s_1 };

	// Linear Zeeman shift
	const myFloat2 Bxy = INV_SQRT_2 * myFloat2{ B.x, B.y };
	const myFloat2 BxyConj = conj(Bxy);
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

__global__ void forwardEuler(PitchedPtr nextStep, PitchedPtr prevStep, int4* __restrict__ laplace, myFloat* __restrict__ hodges, MagFields Bs, uint3 dimensions, myFloat block_scale, myFloat3 p0, myFloat c0, myFloat c2, myFloat alpha, bool USE_THREE_BODY_LOSS, bool USE_QUADRATIC_ZEEMAN, bool USE_QUADRUPOLE_OFFSET, myFloat dt, const myFloat t)
{};
#else
__global__ void forwardEuler(PitchedPtr nextStep, PitchedPtr prevStep, int4* __restrict__ laplace, myFloat* __restrict__ hodges, MagFields Bs, uint3 dimensions, myFloat block_scale, myFloat3 p0, myFloat c0, myFloat c2, myFloat alpha, bool USE_THREE_BODY_LOSS, bool USE_QUADRATIC_ZEEMAN, bool USE_QUADRUPOLE_OFFSET, myFloat dt, const myFloat t)
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

		const myFloat hodge = hodges[primaryFace] / (block_scale * block_scale);
		H.s1 += hodge * (otherBoundaryZeroCell.s1 - prev.s1);
		H.s0 += hodge * (otherBoundaryZeroCell.s0 - prev.s0);
		H.s_1 += hodge * (otherBoundaryZeroCell.s_1 - prev.s_1);

		primaryFace++;
	}

	const myFloat normSq_s1 = prev.s1.x * prev.s1.x + prev.s1.y * prev.s1.y;
	const myFloat normSq_s_1 = prev.s_1.x * prev.s_1.x + prev.s_1.y * prev.s_1.y;
	const myFloat normSq = normSq_s1 + (prev.s0.x * prev.s0.x + prev.s0.y * prev.s0.y) + normSq_s_1;

	const myFloat3 localPos = d_localPos[dualNodeId];
	const myFloat3 globalPos = { p0.x + block_scale * (dataXid * BLOCK_WIDTH_X + localPos.x),
		p0.y + block_scale * (yid * BLOCK_WIDTH_Y + localPos.y),
		p0.z + block_scale * (zid * BLOCK_WIDTH_Z + localPos.z) };

	myFloat2 totalPot = { trap(globalPos, t) + c0 * normSq, 0 };
	if (USE_THREE_BODY_LOSS)
	{
		totalPot.y = -alpha * normSq * normSq;
	}

	H.s1 += totalPot * prev.s1;
	H.s0 += totalPot * prev.s0;
	H.s_1 += totalPot * prev.s_1;

	const myFloat2 magXY = SQRT_2 * (conj(prev.s1) * prev.s0 + conj(prev.s0) * prev.s_1);
	myFloat3 B = magneticField(globalPos, Bs.Bq, Bs.Bb, USE_QUADRUPOLE_OFFSET);
	B += c2 * myFloat3{ magXY.x, magXY.y, normSq_s1 - normSq_s_1 };

	// Linear Zeeman shift
	myFloat2 Bxy = INV_SQRT_2 * myFloat2{ B.x, B.y };
	myFloat2 BxyConj = conj(Bxy);
	H.s1 += (B.z * prev.s1 + BxyConj * prev.s0);
	H.s0 += (Bxy * prev.s1 + BxyConj * prev.s_1);
	H.s_1 += (Bxy * prev.s0 - B.z * prev.s_1);

	if (USE_QUADRATIC_ZEEMAN)
	{
		// Quadratic Zeeman shift
		B = magneticField(globalPos, Bs.BqQuad, Bs.BbQuad, USE_QUADRUPOLE_OFFSET);
		Bxy = INV_SQRT_2 * myFloat2{ B.x, B.y };
		BxyConj = conj(Bxy);
		myFloat BxyNormSq = (BxyConj * Bxy).x;
		myFloat2 BxySq = Bxy * Bxy;
		myFloat2 BxyConjSq = BxyConj * BxyConj;
		myFloat BzSq = B.z * B.z;
		myFloat2 BzBxy = B.z * Bxy;
		myFloat2 BzBxyConj = B.z * BxyConj;
		H.s1 += (BzSq + BxyNormSq) * prev.s1 + BzBxyConj * prev.s0 + BxyConjSq * prev.s_1;
		H.s0 += BzBxy * prev.s1 + 2 * BxyNormSq * prev.s0 - BzBxyConj * prev.s_1;
		H.s_1 += BxySq * prev.s1 - BzBxy * prev.s0 + (BzSq + BxyNormSq) * prev.s_1;
	}

	nextPsi->values[dualNodeId].s1 = prev.s1 + dt * myFloat2{ H.s1.y, -H.s1.x };
	nextPsi->values[dualNodeId].s0 = prev.s0 + dt * myFloat2{ H.s0.y, -H.s0.x };
	nextPsi->values[dualNodeId].s_1 = prev.s_1 + dt * myFloat2{ H.s_1.y, -H.s_1.x };
};

__global__ void leapfrog(PitchedPtr nextStep, PitchedPtr prevStep, const int4* __restrict__ laplace, const myFloat* __restrict__ hodges, MagFields Bs, const uint3 dimensions, const myFloat block_scale, const myFloat3 p0, const myFloat c0, const myFloat c2, myFloat alpha, bool USE_THREE_BODY_LOSS, bool USE_QUADRATIC_ZEEMAN, bool USE_QUADRUPOLE_OFFSET, myFloat dt, const myFloat t)
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

		const myFloat hodge = hodges[primaryFace] / (block_scale * block_scale);
		H.s1 += hodge * (otherBoundaryZeroCell.s1 - prev.s1);
		H.s0 += hodge * (otherBoundaryZeroCell.s0 - prev.s0);
		H.s_1 += hodge * (otherBoundaryZeroCell.s_1 - prev.s_1);

		primaryFace++;
	}

	const myFloat normSq_s1 = prev.s1.x * prev.s1.x + prev.s1.y * prev.s1.y;
	const myFloat normSq_s_1 = prev.s_1.x * prev.s_1.x + prev.s_1.y * prev.s_1.y;
	const myFloat normSq = normSq_s1 + (prev.s0.x * prev.s0.x + prev.s0.y * prev.s0.y) + normSq_s_1;

	const myFloat3 localPos = d_localPos[dualNodeId];
	const myFloat3 globalPos = { p0.x + block_scale * (dataXid * BLOCK_WIDTH_X + localPos.x),
		p0.y + block_scale * (yid * BLOCK_WIDTH_Y + localPos.y),
		p0.z + block_scale * (zid * BLOCK_WIDTH_Z + localPos.z) };

	myFloat2 totalPot = { trap(globalPos, t) + c0 * normSq, 0 };
	if (USE_THREE_BODY_LOSS)
	{
		totalPot.y = -alpha * normSq * normSq;
	}

	const myFloat2 magXY = SQRT_2 * (conj(prev.s1) * prev.s0 + conj(prev.s0) * prev.s_1);
	myFloat3 B = magneticField(globalPos, Bs.Bq, Bs.Bb, USE_QUADRUPOLE_OFFSET);
	B += c2 * myFloat3{ magXY.x, magXY.y, normSq_s1 - normSq_s_1 };

	// Linear Zeeman shift
	myFloat2 Bxy = INV_SQRT_2 * myFloat2{ B.x, B.y };
	myFloat2 BxyConj = conj(Bxy);

	H.s1 += (totalPot + myFloat2{ B.z, 0 }) * prev.s1 + BxyConj * prev.s0;
	H.s0 += Bxy * prev.s1 + totalPot * prev.s0 + BxyConj * prev.s_1;
	H.s_1 += Bxy * prev.s0 + (totalPot - myFloat2{ B.z, 0 }) * prev.s_1;

	if (USE_QUADRATIC_ZEEMAN)
	{
		// Quadratic Zeeman shift
		B = magneticField(globalPos, Bs.BqQuad, Bs.BbQuad, USE_QUADRUPOLE_OFFSET);
		Bxy = INV_SQRT_2 * myFloat2{ B.x, B.y };
		BxyConj = conj(Bxy);
		myFloat BxyNormSq = (BxyConj * Bxy).x;
		myFloat2 BxySq = Bxy * Bxy;
		myFloat2 BxyConjSq = BxyConj * BxyConj;
		myFloat BzSq = B.z * B.z;
		myFloat2 BzBxy = B.z * Bxy;
		myFloat2 BzBxyConj = B.z * BxyConj;
		H.s1 += (BzSq + BxyNormSq) * prev.s1 + BzBxyConj * prev.s0 + BxyConjSq * prev.s_1;
		H.s0 += BzBxy * prev.s1 + 2 * BxyNormSq * prev.s0 - BzBxyConj * prev.s_1;
		H.s_1 += BxySq * prev.s1 - BzBxy * prev.s0 + (BzSq + BxyNormSq) * prev.s_1;
	}

	nextPsi->values[dualNodeId].s1 += 2 * dt * myFloat2{ H.s1.y, -H.s1.x };
	nextPsi->values[dualNodeId].s0 += 2 * dt * myFloat2{ H.s0.y, -H.s0.x };
	nextPsi->values[dualNodeId].s_1 += 2 * dt * myFloat2{ H.s_1.y, -H.s_1.x };
};

__global__ void analyticStep(PitchedPtr nextStep, PitchedPtr prevStep, uint3 dimensions, const myFloat2 phaseShift)
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

void printDensity(dim3 dimGrid, dim3 dimBlock, myFloat* densityPtr, PitchedPtr psi, uint3 dimensions, size_t bodies, myFloat volume)
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

	std::cout << "Total density: " << hDensity << std::endl;
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

myFloat getMaxHamilton(dim3 dimGrid, dim3 dimBlock, myFloat* maxHamlPtr, PitchedPtr psi, MagFields Bs, uint3 dimensions, size_t bodies, myFloat block_scale, myFloat3 p0)
{
	maxHamilton << <dimGrid, dimBlock >> > (maxHamlPtr, psi, Bs, dimensions, block_scale, p0, c0, c2, alpha, t);
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
		std::cout << "Loading data from " << filename << "..." << std::endl;
	}
	psi_fs.read(dst, size);
	psi_fs.close();
}

uint integrateInTime(const myFloat block_scale, const Vector3& minp, const Vector3& maxp)
{
	// find dimensions
	const Vector3 domain = maxp - minp;
	const uint xsize = uint(domain.x / (block_scale * BLOCK_WIDTH.x)); // + 1;
	const uint ysize = uint(domain.y / (block_scale * BLOCK_WIDTH.y)); // + 1;
	const uint zsize = uint(domain.z / (block_scale * BLOCK_WIDTH.z)); // + 1;
	const Vector3 p0 = 0.5 * (minp + maxp - block_scale * Vector3(BLOCK_WIDTH.x * xsize, BLOCK_WIDTH.y * ysize, BLOCK_WIDTH.z * zsize));
	const myFloat3 d_p0 = { p0.x, p0.y, p0.z };

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

	cudaPitchedPtr d_cudaEvenPsi = allocDevice3D(psiExtent);
	cudaPitchedPtr d_cudaOddPsi = allocDevice3D(psiExtent);
	cudaPitchedPtr d_cudaGroundPsi = allocDevice3D(psiExtent);
	cudaPitchedPtr d_cudaAnalyticPsi = allocDevice3D(psiExtent);

	cudaPitchedPtr d_cudaHPsi = allocDevice3D(psiExtent);

	myFloat* d_spinNorm = allocDevice<myFloat>(bodies);
	myFloat* d_density = allocDevice<myFloat>(bodies);
	myFloat* d_energy = allocDevice<myFloat>(bodies);
	myFloat3* d_localAvgSpin = allocDevice<myFloat3>(bodies);
	myFloat3* d_u = allocDevice<myFloat3>(bodies);
	myFloat3* d_v = allocDevice<myFloat3>(bodies);
	myFloat* d_theta = allocDevice<myFloat>(bodies);
	myFloat* d_error = allocDevice<myFloat>(bodies);

	size_t offset = d_cudaEvenPsi.pitch * dysize + d_cudaEvenPsi.pitch + sizeof(BlockPsis);
	PitchedPtr d_evenPsi = { (char*)d_cudaEvenPsi.ptr + offset, d_cudaEvenPsi.pitch, d_cudaEvenPsi.pitch * dysize };
	PitchedPtr d_oddPsi = { (char*)d_cudaOddPsi.ptr + offset, d_cudaOddPsi.pitch, d_cudaOddPsi.pitch * dysize };
	PitchedPtr d_groundPsi = { (char*)d_cudaGroundPsi.ptr + offset, d_cudaGroundPsi.pitch, d_cudaGroundPsi.pitch * dysize };
	PitchedPtr d_analyticPsi = { (char*)d_cudaAnalyticPsi.ptr + offset, d_cudaAnalyticPsi.pitch, d_cudaAnalyticPsi.pitch * dysize };

	PitchedPtr d_HPsi = { (char*)d_cudaHPsi.ptr + offset, d_cudaHPsi.pitch, d_cudaHPsi.pitch * dysize };

	// find terms for laplacian
	Buffer<int4> lapind;
	Buffer<myFloat> hodges;
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

	int4* d_lapind = allocDevice<int4>(lapind.size());
	myFloat* d_hodges = allocDevice<myFloat>(hodges.size());

	// Initialize host memory
	size_t hostSize = dxsize * dysize * dzsize;
	BlockPsis* h_evenPsi = allocHost<BlockPsis>(hostSize);
	BlockPsis* h_oddPsi = allocHost<BlockPsis>(hostSize);
	BlockPsis* h_analyticPsi = allocHost<BlockPsis>(hostSize);
	checkCudaErrors(cudaMallocHost(&h_evenPsi, hostSize * sizeof(BlockPsis)));
	checkCudaErrors(cudaMallocHost(&h_oddPsi, hostSize * sizeof(BlockPsis)));
	checkCudaErrors(cudaMallocHost(&h_analyticPsi, hostSize * sizeof(BlockPsis)));

	myFloat* h_density = allocHost<myFloat>(bodies);
	myFloat3* h_u = allocHost<myFloat3>(bodies);
	myFloat* h_theta = allocHost<myFloat>(bodies);
	myFloat3* h_localAvgSpin = allocHost<myFloat3>(bodies);

#if COMPUTE_GROUND_STATE
	// Initialize discrete field
	std::ifstream fs(GROUND_STATE_FILENAME, std::ios::binary | std::ios::in);
	if (fs.fail() != 0)
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
							const myFloat2 rand_s1 = { distribution(generator), distribution(generator) };
							const myFloat2 rand_s0 = { distribution(generator), distribution(generator) };
							const myFloat2 rand_s_1 = { distribution(generator), distribution(generator) };

							const myFloat dens_s1 = (conj(h_oddPsi[dstI].values[l].s1) * h_oddPsi[dstI].values[l].s1).x;
							const myFloat dens_s0 = (conj(h_oddPsi[dstI].values[l].s0) * h_oddPsi[dstI].values[l].s0).x;
							const myFloat dens_s_1 = (conj(h_oddPsi[dstI].values[l].s_1) * h_oddPsi[dstI].values[l].s_1).x;
							const myFloat dens = dens_s1 + dens_s0 + dens_s_1;

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
	cudaPitchedPtr h_cudaAnalyticPsi = { 0 };

	h_cudaEvenPsi.ptr = h_evenPsi;
	h_cudaEvenPsi.pitch = dxsize * sizeof(BlockPsis);
	h_cudaEvenPsi.xsize = d_cudaEvenPsi.xsize;
	h_cudaEvenPsi.ysize = d_cudaEvenPsi.ysize;

	h_cudaOddPsi.ptr = h_oddPsi;
	h_cudaOddPsi.pitch = dxsize * sizeof(BlockPsis);
	h_cudaOddPsi.xsize = d_cudaOddPsi.xsize;
	h_cudaOddPsi.ysize = d_cudaOddPsi.ysize;

	h_cudaAnalyticPsi.ptr = h_analyticPsi;
	h_cudaAnalyticPsi.pitch = dxsize * sizeof(BlockPsis);
	h_cudaAnalyticPsi.xsize = d_cudaAnalyticPsi.xsize;
	h_cudaAnalyticPsi.ysize = d_cudaAnalyticPsi.ysize;

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
	evenPsiBackParams.srcPtr = d_cudaEvenPsi;
	evenPsiBackParams.dstPtr = h_cudaEvenPsi;
	evenPsiBackParams.extent = psiExtent;
	evenPsiBackParams.kind = cudaMemcpyDeviceToHost;

	cudaMemcpy3DParms oddPsiBackParams = { 0 };
	oddPsiBackParams.srcPtr = d_cudaOddPsi;
	oddPsiBackParams.dstPtr = h_cudaOddPsi;
	oddPsiBackParams.extent = psiExtent;
	oddPsiBackParams.kind = cudaMemcpyDeviceToHost;

	cudaMemcpy3DParms analyticPsiBackParams = { 0 };
	analyticPsiBackParams.srcPtr = d_cudaAnalyticPsi;
	analyticPsiBackParams.dstPtr = h_cudaAnalyticPsi;
	analyticPsiBackParams.extent = psiExtent;
	analyticPsiBackParams.kind = cudaMemcpyDeviceToHost;

	// Integrate in time
	uint3 dimensions = make_uint3(xsize, ysize, zsize);
	dim3 dimBlock(THREAD_BLOCK_X * VALUES_IN_BLOCK, THREAD_BLOCK_Y, THREAD_BLOCK_Z);
	dim3 dimGrid((xsize + THREAD_BLOCK_X - 1) / THREAD_BLOCK_X,
		(ysize + THREAD_BLOCK_Y - 1) / THREAD_BLOCK_Y,
		((zsize + THREAD_BLOCK_Z - 1) / THREAD_BLOCK_Z));

	Signal signal;
	MagFields Bs{};

	const myFloat volume = block_scale * block_scale * block_scale * VOLUME;

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

	cudaMemcpy3DParms groundPsiParams = { 0 };
	groundPsiParams.srcPtr = d_cudaOddPsi;
	groundPsiParams.dstPtr = d_cudaGroundPsi;
	groundPsiParams.extent = psiExtent;
	groundPsiParams.kind = cudaMemcpyDeviceToDevice;
	checkCudaErrors(cudaMemcpy3D(&groundPsiParams));

	constexpr myFloat E = 127.295; // Computed with ITP

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
	}
	else
	{
		std::cout << "Skipping the forward step." << std::endl;
	}

#if COMPUTE_GROUND_STATE
	std::string folder = "gs_dens_profiles_" + EXTRA_INFORMATION;
	std::string createResultsDirCommand = "mkdir " + folder;
	system(createResultsDirCommand.c_str());

	uint iter = 0;

	normalize_h(dimGrid, dimBlock, d_density, d_evenPsi, dimensions, bodies, volume);

	while (true)
	{
		constexpr int ITERS_PER_IMAGE = 100;
		if ((iter % ITERS_PER_IMAGE) == 0) std::cout << "Iteration " << iter << std::endl;
#if SAVE_PICTURE
		if ((iter % ITERS_PER_IMAGE) == 0)
		{
			checkCudaErrors(cudaMemcpy3D(&evenPsiBackParams));
			drawIandR(folder, h_evenPsi, dxsize, dysize, dzsize, iter, Bs, d_p0, block_scale);
			printDensity(dimGrid, dimBlock, d_density, d_evenPsi, dimensions, bodies, volume);

			myFloat3 com = centerOfMass(h_evenPsi, bsize, dxsize, dysize, dzsize, block_scale, d_p0);
			std::cout << "Center of mass: " << com.x << ", " << com.y << ", " << com.z << std::endl;

#if 1
			// Compute energy/chemical potential // Alice ring experimen E = 127.295
			innerProduct << <dimGrid, dimBlock >> > (d_energy, d_evenPsi, d_HPsi, dimensions);
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
	std::string spinorVtksDir = getProjectionString() + "\\spinor_vtks_" + std::to_string(POLAR_FERRO_MIX);
	std::string datsDir = getProjectionString() + "\\dats_" + std::to_string(POLAR_FERRO_MIX);

	std::string createResultsDirCommand = "mkdir " + resultsDir;
	std::string createVtksDirCommand = "mkdir " + vtksDir;
	std::string createSpinorVtksDirCommand = "mkdir " + spinorVtksDir;
	std::string createDatsDirCommand = "mkdir " + datsDir;
	system(createResultsDirCommand.c_str());
	system(createVtksDirCommand.c_str());
	system(createSpinorVtksDirCommand.c_str());
	system(createDatsDirCommand.c_str());

	myFloat expansionBlockScale = block_scale;

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
	//cudaMemcpy(h_u, d_u, bodies * sizeof(myFloat3), cudaMemcpyDeviceToHost);
	//cudaMemcpy(h_theta, d_theta, bodies * sizeof(myFloat), cudaMemcpyDeviceToHost);
	//drawUtheta(h_u, h_theta, xsize, ysize, zsize, t - 202.03);
	//
	//ferromagneticDomain << <dimGrid, dimBlock >> > (d_ferroDom, d_oddPsi, dimensions);
	//cudaMemcpy(h_ferroDom, d_ferroDom, bodies * sizeof(myFloat), cudaMemcpyDeviceToHost);
	//drawFerroDom(h_ferroDom, xsize, ysize, zsize, t - 202.03);
#endif

	myFloat phaseTime = 0;

	while (t < END_TIME)
	{
		// integrate one iteration
		for (uint step = 0; step < IMAGE_SAVE_FREQUENCY; step++)
		{
			// update odd values
			phaseTime += dt;
			t += dt / omega_r * 1e3; // [ms]
			if (t >= EXPANSION_START) {
				myFloat k = 0.82; // 0.7569772335291065; // From the Aalto QCD code for F=2
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
				myFloat k = 0.82; // 0.7569772335291065; // From the Aalto QCD code for F=2
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
		// Measure wall clock time
		auto duration = std::chrono::high_resolution_clock::now() - prevTime;
		//std::cout << "Simulation time: " << t << " ms. Real time from previous save: " << duration.count() * 1e-9 << " s." << std::endl;
		prevTime = std::chrono::high_resolution_clock::now();

		myFloat2 phaseShift = myFloat2{ cos(-phaseTime * E), sin(-phaseTime * E) };
		analyticStep << <dimGrid, dimBlock >> > (d_analyticPsi, d_groundPsi, dimensions, phaseShift);
		checkCudaErrors(cudaMemcpy3D(&analyticPsiBackParams));
		drawDensityRI("analytic_", h_analyticPsi, dxsize, dysize, dzsize, t - CREATION_RAMP_START, resultsDir);

		// Copy back from device memory to host memory
		checkCudaErrors(cudaMemcpy3D(&oddPsiBackParams));
		drawDensityRI("", h_oddPsi, dxsize, dysize, dzsize, t - CREATION_RAMP_START, resultsDir);

		// Compute error
		{
			weightedDiff << <dimGrid, dimBlock >> > (d_error, d_analyticPsi, d_oddPsi, dimensions);
			int prevStride = bodies;
			while (prevStride > 1)
			{
				int newStride = prevStride / 2;
				integrate << <dim3(std::ceil(newStride / 32.0), 1, 1), dim3(32, 1, 1) >> > (d_error, newStride, ((newStride * 2) != prevStride), volume);
				prevStride = newStride;
			}
			myFloat hError = { 0 };
			checkCudaErrors(cudaMemcpy(&hError, d_error, sizeof(myFloat), cudaMemcpyDeviceToHost));
			std::cout << hError << ", ";
		}

		//uvTheta << <dimGrid, dimBlock >> > (d_u, d_v, d_theta, d_oddPsi, dimensions);
		//cudaMemcpy(h_u, d_u, bodies * sizeof(myFloat3), cudaMemcpyDeviceToHost);
		//cudaMemcpy(h_theta, d_theta, bodies * sizeof(myFloat), cudaMemcpyDeviceToHost);
		//drawUtheta(h_u, h_theta, xsize, ysize, zsize, t - 202.03);
		//
		//ferromagneticDomain << <dimGrid, dimBlock >> > (d_ferroDom, d_oddPsi, dimensions);
		//cudaMemcpy(h_ferroDom, d_ferroDom, bodies * sizeof(myFloat), cudaMemcpyDeviceToHost);
		//drawFerroDom(h_ferroDom, xsize, ysize, zsize, t - 202.03);
#endif
#if SAVE_STATES
		// Copy back from device memory to host memory
		checkCudaErrors(cudaMemcpy3D(&oddPsiBackParams));

		localAvgSpinAndDensity << <dimGrid, dimBlock >> > (d_spinNorm, d_localAvgSpin, d_density, d_oddPsi, dimensions);
		cudaMemcpy(h_localAvgSpin, d_localAvgSpin, bodies * sizeof(myFloat3), cudaMemcpyDeviceToHost);
		uvTheta << <dimGrid, dimBlock >> > (d_u, d_v, d_theta, d_oddPsi, dimensions);
		cudaMemcpy(h_u, d_u, bodies * sizeof(myFloat3), cudaMemcpyDeviceToHost);
		cudaMemcpy(h_theta, d_theta, bodies * sizeof(myFloat), cudaMemcpyDeviceToHost);
		
		//saveVolume(SAVE_FILE_PREFIX, h_oddPsi, h_localAvgSpin, h_u, h_theta, bsize, dxsize, dysize, dzsize, 0, block_scale, d_p0, t - CREATION_RAMP_START, vtksDir);
		//saveSpinor(spinorVtksDir, h_oddPsi, bsize, dxsize, dysize, dzsize, block_scale, d_p0, t - CREATION_RAMP_START);
		savePreImageSpinor(spinorVtksDir, h_oddPsi, bsize, dxsize, dysize, dzsize, block_scale, d_p0, t - CREATION_RAMP_START);

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

myFloat tau = 0.01;
myFloat dt_per_tau = dt / tau;

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
				dt_per_tau = dt / tau;
			}
			else if (size_t pos = line.find("tau") != std::string::npos)
			{
				tau = std::stod(line.substr(pos + 5));
				dt_per_tau = dt / tau;
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

	const myFloat targetBlockWidth = DOMAIN_SIZE_X / REPLICABLE_STRUCTURE_COUNT_X;
	const myFloat blockScale = targetBlockWidth / BLOCK_WIDTH_X;

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
