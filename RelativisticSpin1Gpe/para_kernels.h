#ifndef PARA_KERNELS_H
#define PARA_KERNELS_H

__global__ void update_q_para(PitchedPtr next_q, PitchedPtr psi, int3* d0, uint3 dimensions)
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

	next->values[dualEdgeId] = d0psi;
}

#endif // PARA_KERNELS_H