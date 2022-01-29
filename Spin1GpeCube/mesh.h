#define FACE_COUNT 6
#define DUAL_EDGE_LENGTH 1
#define VALUES_IN_BLOCK 1
#define INDICES_PER_BLOCK 6
const Vector3 BLOCK_WIDTH = Vector3(1, 1, 1); // dimensions of unit block
__constant__ double BLOCK_WIDTH_X = 1; // dimensions of unit block
__constant__ double BLOCK_WIDTH_Y = 1; // dimensions of unit block
__constant__ double BLOCK_WIDTH_Z = 1; // dimensions of unit block
const ddouble VOLUME = 1; // volume of body elements
const bool IS_3D = true; // 3-dimensional
__host__ __device__ double3 getLocalPos(int dualZeroCellIndex)
{
	double3 pos[VALUES_IN_BLOCK];
	pos[0] = make_double3(0.5, 0.5, 0.5);

	return pos[dualZeroCellIndex];
}
ddouble getLaplacian(Buffer<int2> &ind, Buffer<ddouble> &hodges, const int nx, const int ny, const int nz) // nx, ny, nz in bytes
{
	ind.resize(INDICES_PER_BLOCK);
	ind[0] = make_int2(-nz, 0);
	ind[1] = make_int2(-nx, 0);
	ind[2] = make_int2(-ny, 0);
	ind[3] = make_int2(nx, 0);
	ind[4] = make_int2(ny, 0);
	ind[5] = make_int2(nz, 0);

	hodges.resize(INDICES_PER_BLOCK);
	hodges[0] = 1;
	hodges[1] = 1;
	hodges[2] = 1;
	hodges[3] = 1;
	hodges[4] = 1;
	hodges[5] = 1;

	return 1;
}
