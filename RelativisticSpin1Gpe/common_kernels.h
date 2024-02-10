#ifndef COMMON_KERNELS_H
#define COMMON_KERNELS_H

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

__global__ void itp_psi(PitchedPtr HPsiPtr, PitchedPtr nextStep, PitchedPtr prevStep, PitchedPtr qs, const int2* __restrict__ d1Ptr, const double* __restrict__ hodges, MagFields Bs, const uint3 dimensions, const double block_scale, const double3 p0, const double c0, const double c2, double dt)
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

	// For computing the energy/chemical potential
	BlockPsis* HPsi = (BlockPsis*)(HPsiPtr.ptr + HPsiPtr.slicePitch * zid + HPsiPtr.pitch * yid) + dataXid;

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

__global__ void forwardEuler(PitchedPtr nextStep, PitchedPtr prevStep, PitchedPtr qs, const int2* __restrict__ d1Ptr, const double* __restrict__ hodges, MagFields Bs, const uint3 dimensions, const double block_scale, const double3 p0, const double c0, const double c2, double dt, bool hyperb, double sigma)
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

	// Add the second spatial exterior derivative (d1 of d0) to the Hamiltonian
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

	const double normSq_s1 = prev.s1.x * prev.s1.x + prev.s1.y * prev.s1.y;
	const double normSq_s_1 = prev.s_1.x * prev.s_1.x + prev.s_1.y * prev.s_1.y;
	const double normSq = normSq_s1 + (prev.s0.x * prev.s0.x + prev.s0.y * prev.s0.y) + normSq_s_1;

	const double3 localPos = d_localPos[dualNodeId];
	const double3 globalPos = { p0.x + block_scale * (dataXid * BLOCK_WIDTH_X + localPos.x),
		p0.y + block_scale * (yid * BLOCK_WIDTH_Y + localPos.y),
		p0.z + block_scale * (zid * BLOCK_WIDTH_Z + localPos.z) };

	double2 totalPot = { trap(globalPos) + c0 * normSq, 0 };

	if (hyperb)
	{
		double c_K = 1 + (2 / F) * sigma * totalPot.x;
		H.s1 = -c_K * H.s1;
		H.s0 = -c_K * H.s0;
		H.s_1 = -c_K * H.s_1;
	}

	H.s1 += totalPot * prev.s1;
	H.s0 += totalPot * prev.s0;
	H.s_1 += totalPot * prev.s_1;

	const double2 magXY = SQRT_2 * (conj(prev.s1) * prev.s0 + conj(prev.s0) * prev.s_1);
	double3 B = magneticField(globalPos, Bs.Bq, Bs.Bb);
	B += c2 * double3{ magXY.x, magXY.y, normSq_s1 - normSq_s_1 };

	// Linear Zeeman shift
	double2 Bxy = INV_SQRT_2 * double2{ B.x, B.y };
	double2 BxyConj = conj(Bxy);
	H.s1 += (B.z * prev.s1 + BxyConj * prev.s0);
	H.s0 += (Bxy * prev.s1 + BxyConj * prev.s_1);
	H.s_1 += (Bxy * prev.s0 - B.z * prev.s_1);

	nextPsi->values[dualNodeId].s1 = prev.s1 + dt * double2{ H.s1.y, -H.s1.x };
	nextPsi->values[dualNodeId].s0 = prev.s0 + dt * double2{ H.s0.y, -H.s0.x };
	nextPsi->values[dualNodeId].s_1 = prev.s_1 + dt * double2{ H.s_1.y, -H.s_1.x };
};

__global__ void update_psi(PitchedPtr nextStep, PitchedPtr prevStep, PitchedPtr qs, const int2* __restrict__ d1Ptr, const double* __restrict__ hodges, MagFields Bs, const uint3 dimensions, const double block_scale, const double3 p0, const double c0, const double c2, double dt, bool hyperb, double sigma)
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

	// Add the the second exterior derivative (d1 of d0) to the Hamiltonian
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

	const double normSq_s1 = prev.s1.x * prev.s1.x + prev.s1.y * prev.s1.y;
	const double normSq_s_1 = prev.s_1.x * prev.s_1.x + prev.s_1.y * prev.s_1.y;
	const double normSq = normSq_s1 + (prev.s0.x * prev.s0.x + prev.s0.y * prev.s0.y) + normSq_s_1;

	const double3 localPos = d_localPos[dualNodeId];
	const double3 globalPos = { p0.x + block_scale * (dataXid * BLOCK_WIDTH_X + localPos.x),
		p0.y + block_scale * (yid * BLOCK_WIDTH_Y + localPos.y),
		p0.z + block_scale * (zid * BLOCK_WIDTH_Z + localPos.z) };

	double2 totalPot = { trap(globalPos) + c0 * normSq, 0 };

	if (hyperb)
	{
		double c_K = 1 + (2 / F) * sigma * totalPot.x;
		H.s1 = -c_K * H.s1;
		H.s0 = -c_K * H.s0;
		H.s_1 = -c_K * H.s_1;
	}

	H.s1 += totalPot * prev.s1;
	H.s0 += totalPot * prev.s0;
	H.s_1 += totalPot * prev.s_1;

	const double2 magXY = SQRT_2 * (conj(prev.s1) * prev.s0 + conj(prev.s0) * prev.s_1);
	double3 B = magneticField(globalPos, Bs.Bq, Bs.Bb);
	B += c2 * double3{ magXY.x, magXY.y, normSq_s1 - normSq_s_1 };

	// Linear Zeeman shift
	double2 Bxy = INV_SQRT_2 * double2{ B.x, B.y };
	double2 BxyConj = conj(Bxy);
	H.s1 += (B.z * prev.s1 + BxyConj * prev.s0);
	H.s0 += (Bxy * prev.s1 + BxyConj * prev.s_1);
	H.s_1 += (Bxy * prev.s0 - B.z * prev.s_1);

	nextPsi->values[dualNodeId].s1 += 2.0 * dt * double2{ H.s1.y, -H.s1.x };
	nextPsi->values[dualNodeId].s0 += 2.0 * dt * double2{ H.s0.y, -H.s0.x };
	nextPsi->values[dualNodeId].s_1 += 2.0 * dt * double2{ H.s_1.y, -H.s_1.x };
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

__global__ void maxHamilton(double* maxHamlPtr, PitchedPtr prevStep, MagFields Bs, uint3 dimensions, double block_scale, double3 p0, double c0, double c2)
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
	double3 B = magneticField(globalPos, Bs.Bq, Bs.Bb);
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

__global__ void innerProductReal(double* result, PitchedPtr pLeft, PitchedPtr pRight, uint3 dimensions)
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

__global__ void innerProduct(double2* result, PitchedPtr pLeft, PitchedPtr pRight, uint3 dimensions)
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
	result[idx] = conj(left.s1) * right.s1 + conj(left.s0) * right.s0 + conj(left.s_1) * right.s_1;
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

	Complex3Vec left = ((BlockPsis*)(pLeft.ptr + pLeft.slicePitch * zid + pLeft.pitch * yid) + dataXid)->values[dualNodeId];
	Complex3Vec right = ((BlockPsis*)(pRight.ptr + pRight.slicePitch * zid + pRight.pitch * yid) + dataXid)->values[dualNodeId];

	Complex3Vec diff = { right.s1 - left.s1, right.s0 - left.s0, right.s_1 - left.s_1 };

	double leftSqr = (conj(left.s1) * left.s1).x + (conj(left.s0) * left.s0).x + (conj(left.s_1) * left.s_1).x;
	double diffSqr = (conj(diff.s1) * diff.s1).x + (conj(diff.s0) * diff.s0).x + (conj(diff.s_1) * diff.s_1).x;

	size_t idx = VALUES_IN_BLOCK * (zid * dimensions.x * dimensions.y + yid * dimensions.x + dataXid) + dualNodeId;
	//result[idx] = leftSqr * diffSqr;
	result[idx] = diffSqr;
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

__global__ void com(double3* com, PitchedPtr prevStep, uint3 dimensions, const double block_scale, const double3 p0)
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

	double dens = (psi.s1 * conj(psi.s1)).x + (psi.s0 * conj(psi.s0)).x + (psi.s_1 * conj(psi.s_1)).x;
	const double3 localPos = d_localPos[dualNodeId];
	const double3 globalPos = { p0.x + block_scale * (dataXid * BLOCK_WIDTH_X + localPos.x),
		p0.y + block_scale * (yid * BLOCK_WIDTH_Y + localPos.y),
		p0.z + block_scale * (zid * BLOCK_WIDTH_Z + localPos.z) };

	size_t idx = VALUES_IN_BLOCK * (zid * dimensions.x * dimensions.y + yid * dimensions.x + dataXid) + dualNodeId;
	com[idx] = dens * globalPos;
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

__global__ void integrate(double2* dataVec, size_t stride, bool addLast, double dv)
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

#endif // COMMON_KERNELS_H