#define FACE_COUNT 4
#define DUAL_EDGE_LENGTH 0.80256619664673123
#define VALUES_IN_BLOCK 12
#define EDGES_IN_BLOCK 24
#define INDICES_PER_BLOCK 48
const Vector3 BLOCK_WIDTH = Vector3(2.27, 2.27, 2.27); // dimensions of unit block
const ddouble VOLUME = 0.97475691666666664; // volume of body elements
const bool IS_3D = true; // 3-dimensional
void getPositions(Buffer<Vector3> &pos)
{
	pos.resize(VALUES_IN_BLOCK);
	pos[0] = Vector3(0.28374999999999967, 0.85124999999999962, 1.9862500000000001);
	pos[1] = Vector3(1.9862500000000001, 0.28374999999999967, 0.85124999999999962);
	pos[2] = Vector3(1.9862500000000001, 0.85124999999999962, 0.28374999999999967);
	pos[3] = Vector3(1.9862500000000001, 0.85124999999999962, 1.4187499999999997);
	pos[4] = Vector3(1.9862500000000001, 1.4187499999999997, 0.85124999999999962);
	pos[5] = Vector3(0.28374999999999967, 1.9862500000000001, 0.85124999999999962);
	pos[6] = Vector3(0.85124999999999962, 1.9862500000000001, 0.28374999999999967);
	pos[7] = Vector3(0.85124999999999962, 1.9862500000000001, 1.4187499999999997);
	pos[8] = Vector3(1.4187499999999997, 1.9862500000000001, 0.85124999999999962);
	pos[9] = Vector3(0.85124999999999962, 0.28374999999999967, 1.9862500000000001);
	pos[10] = Vector3(0.85124999999999962, 1.4187499999999997, 1.9862500000000001);
	pos[11] = Vector3(1.4187499999999997, 0.85124999999999962, 1.9862500000000001);
}
ddouble getLaplacian(Buffer<ddouble>& hodges, Buffer<int3>& edges, Buffer<int2>& edgeInds, const int nx, const int ny, const int nz) // nx, ny, nz in bytes
{
	edges.resize(EDGES_IN_BLOCK);

	edgeInds.resize(VALUES_IN_BLOCK);

	edges[0] = {make_int3(0, 0, 9)};
	edges[1] = {make_int3(0, 0, 10)};
	edges[2] = {make_int3(0, -nx + nz, 2)};
	edges[3] = {make_int3(0, -nx, 3)};
	edges[4] = {make_int3(1, 0, 2)};
	edges[5] = {make_int3(1, 0, 3)};
	edges[6] = {make_int3(1, nx - ny, 5)};
	edges[7] = {make_int3(1, -ny, 8)};
	edges[8] = {make_int3(2, 0, 4)};
	edges[9] = {make_int3(2, -nz, 11)};
	edges[10] = {make_int3(3, 0, 4)};
	edges[11] = {make_int3(3, 0, 11)};
	edges[12] = {make_int3(4, nx, 5)};
	edges[13] = {make_int3(4, 0, 8)};
	edges[14] = {make_int3(5, 0, 6)};
	edges[15] = {make_int3(5, 0, 7)};
	edges[16] = {make_int3(6, 0, 8)};
	edges[17] = {make_int3(6, ny - nz, 9)};
	edges[18] = {make_int3(6, -nz, 10)};
	edges[19] = {make_int3(7, 0, 8)};
	edges[20] = {make_int3(7, ny, 9)};
	edges[21] = {make_int3(7, 0, 10)};
	edges[22] = {make_int3(9, 0, 11)};
	edges[23] = {make_int3(10, 0, 11)};

	//0
	edgeInds[0] = make_int2(0, 0);
	edgeInds[1] = make_int2(0, 1);
	edgeInds[2] = make_int2(0, 2);
	edgeInds[3] = make_int2(0, 3);
	//1
	edgeInds[4] = make_int2(0, 4);
	edgeInds[5] = make_int2(0, 5);
	edgeInds[6] = make_int2(0, 6);
	edgeInds[7] = make_int2(0, 7);
	//2
	edgeInds[8] = make_int2(-(-nx + nz), 2);
	edgeInds[9] = make_int2(-(0), 4);
	edgeInds[10] = make_int2(0, 8);
	edgeInds[11] = make_int2(0, 9);
	//3
	edgeInds[12] = make_int2(-(-nx), 3);
	edgeInds[13] = make_int2(-(0), 5);
	edgeInds[14] = make_int2(0, 10);
	edgeInds[15] = make_int2(0, 11);
	//4
	edgeInds[16] = make_int2(-(0), 8);
	edgeInds[17] = make_int2(-(0), 10);
	edgeInds[18] = make_int2(0, 12);
	edgeInds[19] = make_int2(0, 13);
	//5
	edgeInds[20] = make_int2(-(nx - ny), 6);
	edgeInds[21] = make_int2(-(nx), 12);
	edgeInds[22] = make_int2(0, 14);
	edgeInds[23] = make_int2(0, 15);
	//6
	edgeInds[24] = make_int2(-(0), 14);
	edgeInds[25] = make_int2(0, 16);
	edgeInds[26] = make_int2(0, 17);
	edgeInds[27] = make_int2(0, 18);
	//7
	edgeInds[28] = make_int2(-(0), 15);
	edgeInds[29] = make_int2(0, 19);
	edgeInds[30] = make_int2(0, 20);
	edgeInds[31] = make_int2(0, 21);
	//8
	edgeInds[32] = make_int2(-(-ny), 7);
	edgeInds[33] = make_int2(-(0), 13);
	edgeInds[34] = make_int2(-(0), 16);
	edgeInds[35] = make_int2(-(0), 19);
	//9
	edgeInds[36] = make_int2(-(0), 0);
	edgeInds[37] = make_int2(-(ny - nz), 17);
	edgeInds[38] = make_int2(-(ny), 20);
	edgeInds[39] = make_int2(0, 22);
	//10
	edgeInds[40] = make_int2(-(0), 1);
	edgeInds[41] = make_int2(-(-nz), 18);
	edgeInds[42] = make_int2(-(0), 21);
	edgeInds[43] = make_int2(0, 23);
	//11
	edgeInds[44] = make_int2(-(-nz), 9);
	edgeInds[45] = make_int2(-(0), 11);
	edgeInds[46] = make_int2(-(0), 22);
	edgeInds[47] = make_int2(-(0), 23);

	hodges.resize(INDICES_PER_BLOCK);
	hodges[0] = 2.328785732306081;
	hodges[1] = 2.3287857323060801;
	hodges[2] = 2.3287857323060801;
	hodges[3] = 2.3287857323060797;
	hodges[4] = 2.328785732306081;
	hodges[5] = 2.3287857323060801;
	hodges[6] = 2.3287857323060801;
	hodges[7] = 2.3287857323060797;
	hodges[8] = -2.328785732306081;
	hodges[9] = -2.3287857323060801;
	hodges[10] = 2.3287857323060801;
	hodges[11] = 2.3287857323060797;
	hodges[12] = -2.3287857323060801;
	hodges[13] = -2.3287857323060801;
	hodges[14] = 2.3287857323060801;
	hodges[15] = 2.3287857323060788;
	hodges[16] = -2.3287857323060801;
	hodges[17] = -2.3287857323060801;
	hodges[18] = 2.3287857323060801;
	hodges[19] = 2.3287857323060788;
	hodges[20] = -2.328785732306081;
	hodges[21] = -2.3287857323060801;
	hodges[22] = 2.3287857323060801;
	hodges[23] = 2.3287857323060797;
	hodges[24] = -2.328785732306081;
	hodges[25] = 2.3287857323060801;
	hodges[26] = 2.3287857323060801;
	hodges[27] = 2.3287857323060797;
	hodges[28] = -2.3287857323060801;
	hodges[29] = 2.3287857323060801;
	hodges[30] = 2.3287857323060801;
	hodges[31] = 2.3287857323060788;
	hodges[32] = -2.3287857323060801;
	hodges[33] = -2.3287857323060801;
	hodges[34] = -2.3287857323060801;
	hodges[35] = -2.3287857323060788;
	hodges[36] = -2.328785732306081;
	hodges[37] = -2.3287857323060801;
	hodges[38] = -2.3287857323060801;
	hodges[39] = 2.3287857323060797;
	hodges[40] = -2.3287857323060801;
	hodges[41] = -2.3287857323060801;
	hodges[42] = -2.3287857323060801;
	hodges[43] = 2.3287857323060788;
	hodges[44] = -2.3287857323060801;
	hodges[45] = -2.3287857323060801;
	hodges[46] = -2.3287857323060801;
	hodges[47] = -2.3287857323060788;

	return 2.328785732306081;
}
