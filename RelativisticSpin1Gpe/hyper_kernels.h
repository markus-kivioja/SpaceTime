#ifndef HYPER_KERNELS_H
#define HYPER_KERNELS_H

__global__ void itp_q_hyper(PitchedPtr next_q, PitchedPtr prev_q, PitchedPtr psi, int3* d0, uint3 dimensions, double dt_per_sigma)
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

	BlockEdges* prev = (BlockEdges*)(prev_q.ptr + prev_q.slicePitch * zid + prev_q.pitch * yid) + dataXid;

	Complex3Vec q;
	q.s1 = dt_per_sigma * (-d0psi.s1 + prev->values[dualEdgeId].s1);
	q.s0 = dt_per_sigma * (-d0psi.s0 + prev->values[dualEdgeId].s0);
	q.s_1 = dt_per_sigma * (-d0psi.s_1 + prev->values[dualEdgeId].s_1);

	next->values[dualEdgeId].s1 = prev->values[dualEdgeId].s1 - q.s1;
	next->values[dualEdgeId].s0 = prev->values[dualEdgeId].s0 - q.s0;
	next->values[dualEdgeId].s_1 = prev->values[dualEdgeId].s_1 - q.s_1;
}

__global__ void forwardEuler_q_hyper(PitchedPtr next_q, PitchedPtr prev_q, PitchedPtr psi, int3* d0, uint3 dimensions, double dt_per_sigma)
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

	BlockEdges* prev = (BlockEdges*)(prev_q.ptr + prev_q.slicePitch * zid + prev_q.pitch * yid) + dataXid;

	Complex3Vec q;
	q.s1 = dt_per_sigma * (d0psi.s1 - prev->values[dualEdgeId].s1);
	q.s0 = dt_per_sigma * (d0psi.s0 - prev->values[dualEdgeId].s0);
	q.s_1 = dt_per_sigma * (d0psi.s_1 - prev->values[dualEdgeId].s_1);

	next->values[dualEdgeId].s1 = prev->values[dualEdgeId].s1 + make_double2(q.s1.y, -q.s1.x);
	next->values[dualEdgeId].s0 = prev->values[dualEdgeId].s0 + make_double2(q.s0.y, -q.s0.x);
	next->values[dualEdgeId].s_1 = prev->values[dualEdgeId].s_1 + make_double2(q.s_1.y, -q.s_1.x);
}

__global__ void update_q_hyper(PitchedPtr next_q, PitchedPtr prev_q, PitchedPtr psi, int3* d0, uint3 dimensions, double dt_per_sigma)
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

	BlockEdges* prev = (BlockEdges*)(prev_q.ptr + prev_q.slicePitch * zid + prev_q.pitch * yid) + dataXid;

	Complex3Vec q;
	q.s1 = 2 * dt_per_sigma * (d0psi.s1 - prev->values[dualEdgeId].s1);
	q.s0 = 2 * dt_per_sigma * (d0psi.s0 - prev->values[dualEdgeId].s0);
	q.s_1 = 2 * dt_per_sigma * (d0psi.s_1 - prev->values[dualEdgeId].s_1);

	next->values[dualEdgeId].s1 += make_double2(q.s1.y, -q.s1.x);
	next->values[dualEdgeId].s0 += make_double2(q.s0.y, -q.s0.x);
	next->values[dualEdgeId].s_1 += make_double2(q.s_1.y, -q.s_1.x);
}

#endif // HYPER_KERNELS_H