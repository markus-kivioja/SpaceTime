#define FACE_COUNT 6
#define DUAL_EDGE_LENGTH 1
#define VALUES_IN_BLOCK 1
#define EDGES_IN_BLOCK 3
#define INDICES_PER_BLOCK 6
const Vector3 BLOCK_WIDTH = Vector3(1, 1, 1); // dimensions of unit block
const ddouble VOLUME = 1; // volume of body elements
const bool IS_3D = true; // 3-dimensional
void getPositions(Buffer<Vector3>& pos)
{
	pos.resize(VALUES_IN_BLOCK);
	pos[0] = Vector3(0.5, 0.5, 0.5);
}
ddouble getLaplacian(Buffer<ddouble>& hodges, Buffer<int3>& d0, Buffer<int2>& d1, const int d0x, const int d0y, const int d0z, const int d1x, const int d1y, const int d1z) // offsets in bytes
{
	d0.resize(EDGES_IN_BLOCK);
	d0[0] = { make_int3(0, -d0z, 0) };
	d0[1] = { make_int3(0, -d0x, 0) };
	d0[2] = { make_int3(0, -d0y, 0) };

	d1.resize(INDICES_PER_BLOCK);
	//0
	d1[0] = make_int2(0, 0);
	d1[1] = make_int2(-(-d1z), 0);
	d1[2] = make_int2(0, 1);
	d1[3] = make_int2(-(-d1x), 1);
	d1[4] = make_int2(0, 2);
	d1[5] = make_int2(-(-d1y), 2);

	hodges.resize(INDICES_PER_BLOCK);
	hodges[0] = 1;
	hodges[1] = -1;
	hodges[2] = 1;
	hodges[3] = -1;
	hodges[4] = 1;
	hodges[5] = -1;

	return 1;
}
