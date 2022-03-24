#ifndef MESH_H
#define MESH_H

#define FACE_COUNT 4
#define DUAL_EDGE_LENGTH 0.80256619664673123
#define VALUES_IN_BLOCK 12
#define INDICES_PER_BLOCK 48
const Vector3 BLOCK_WIDTH = Vector3(2.27, 2.27, 2.27); // dimensions of unit block
constexpr double BLOCK_WIDTH_X = 2.27; // dimensions of unit block
constexpr double BLOCK_WIDTH_Y = 2.27; // dimensions of unit block
constexpr double BLOCK_WIDTH_Z = 2.27; // dimensions of unit block
const ddouble VOLUME = 0.97475691666666664; // volume of body elements
const bool IS_3D = true; // 3-dimensional

__constant__ double3 d_localPos[VALUES_IN_BLOCK] = {
{0.28374999999999967, 0.85124999999999962, 1.9862500000000001},
{1.9862500000000001, 0.28374999999999967, 0.85124999999999962},
{1.9862500000000001, 0.85124999999999962, 0.28374999999999967},
{1.9862500000000001, 0.85124999999999962, 1.4187499999999997},
{1.9862500000000001, 1.4187499999999997, 0.85124999999999962},
{0.28374999999999967, 1.9862500000000001, 0.85124999999999962},
{0.85124999999999962, 1.9862500000000001, 0.28374999999999967},
{0.85124999999999962, 1.9862500000000001, 1.4187499999999997},
{1.4187499999999997, 1.9862500000000001, 0.85124999999999962},
{0.85124999999999962, 0.28374999999999967, 1.9862500000000001},
{0.85124999999999962, 1.4187499999999997, 1.9862500000000001},
{1.4187499999999997, 0.85124999999999962, 1.9862500000000001} };

double3 getLocalPos(int dualZeroCellIndex)
{
	double3 pos[VALUES_IN_BLOCK];
	pos[0] = make_double3(0.28374999999999967, 0.85124999999999962, 1.9862500000000001);
	pos[1] = make_double3(1.9862500000000001, 0.28374999999999967, 0.85124999999999962);
	pos[2] = make_double3(1.9862500000000001, 0.85124999999999962, 0.28374999999999967);
	pos[3] = make_double3(1.9862500000000001, 0.85124999999999962, 1.4187499999999997);
	pos[4] = make_double3(1.9862500000000001, 1.4187499999999997, 0.85124999999999962);
	pos[5] = make_double3(0.28374999999999967, 1.9862500000000001, 0.85124999999999962);
	pos[6] = make_double3(0.85124999999999962, 1.9862500000000001, 0.28374999999999967);
	pos[7] = make_double3(0.85124999999999962, 1.9862500000000001, 1.4187499999999997);
	pos[8] = make_double3(1.4187499999999997, 1.9862500000000001, 0.85124999999999962);
	pos[9] = make_double3(0.85124999999999962, 0.28374999999999967, 1.9862500000000001);
	pos[10] = make_double3(0.85124999999999962, 1.4187499999999997, 1.9862500000000001);
	pos[11] = make_double3(1.4187499999999997, 0.85124999999999962, 1.9862500000000001);

	return pos[dualZeroCellIndex];
}
ddouble getLaplacian(Buffer<int4>& ind, Buffer<ddouble>& hodges, const int nx, const int ny, const int nz) // nx, ny, nz in bytes
{
	ind.resize(INDICES_PER_BLOCK);
	// Dual node idx:  0
	ind[0] = make_int4(0, 0, 0, 9);
	ind[1] = make_int4(0, 0, 0, 10);
	ind[2] = make_int4(-1, 0, 1, 2);
	ind[3] = make_int4(-1, 0, 0, 3);
	// Dual node idx:  1
	ind[4] = make_int4(0, 0, 0, 2);
	ind[5] = make_int4(0, 0, 0, 3);
	ind[6] = make_int4(1, -1, 0, 5);
	ind[7] = make_int4(0, -1, 0, 8);
	// Dual node idx:  2
	ind[8] = make_int4(0, 0, 0, 1);
	ind[9] = make_int4(0, 0, 0, 4);
	ind[10] = make_int4(1, 0, -1, 0);
	ind[11] = make_int4(0, 0, -1, 11);
	// Dual node idx:  3
	ind[12] = make_int4(0, 0, 0, 1);
	ind[13] = make_int4(0, 0, 0, 4);
	ind[14] = make_int4(1, 0, 0, 0);
	ind[15] = make_int4(0, 0, 0, 11);
	// Dual node idx:  4
	ind[16] = make_int4(0, 0, 0, 2);
	ind[17] = make_int4(0, 0, 0, 3);
	ind[18] = make_int4(1, 0, 0, 5);
	ind[19] = make_int4(0, 0, 0, 8);
	// Dual node idx:  5
	ind[20] = make_int4(0, 0, 0, 6);
	ind[21] = make_int4(0, 0, 0, 7);
	ind[22] = make_int4(-1, 1, 0, 1);
	ind[23] = make_int4(-1, 0, 0, 4);
	// Dual node idx:  6
	ind[24] = make_int4(0, 0, 0, 5);
	ind[25] = make_int4(0, 0, 0, 8);
	ind[26] = make_int4(0, 1, -1, 9);
	ind[27] = make_int4(0, 0, -1, 10);
	// Dual node idx:  7
	ind[28] = make_int4(0, 0, 0, 5);
	ind[29] = make_int4(0, 0, 0, 8);
	ind[30] = make_int4(0, 1, 0, 9);
	ind[31] = make_int4(0, 0, 0, 10);
	// Dual node idx:  8
	ind[32] = make_int4(0, 0, 0, 6);
	ind[33] = make_int4(0, 0, 0, 7);
	ind[34] = make_int4(0, 1, 0, 1);
	ind[35] = make_int4(0, 0, 0, 4);
	// Dual node idx:  9
	ind[36] = make_int4(0, 0, 0, 0);
	ind[37] = make_int4(0, 0, 0, 11);
	ind[38] = make_int4(0, -1, 1, 6);
	ind[39] = make_int4(0, -1, 0, 7);
	// Dual node idx:  10
	ind[40] = make_int4(0, 0, 0, 0);
	ind[41] = make_int4(0, 0, 0, 11);
	ind[42] = make_int4(0, 0, 1, 6);
	ind[43] = make_int4(0, 0, 0, 7);
	// Dual node idx:  11
	ind[44] = make_int4(0, 0, 0, 9);
	ind[45] = make_int4(0, 0, 0, 10);
	ind[46] = make_int4(0, 0, 1, 2);
	ind[47] = make_int4(0, 0, 0, 3);

	hodges.resize(INDICES_PER_BLOCK);
	hodges[0] = 2.328785732306081;
	hodges[1] = 2.3287857323060801;
	hodges[2] = 2.3287857323060801;
	hodges[3] = 2.3287857323060797;
	hodges[4] = 2.328785732306081;
	hodges[5] = 2.3287857323060801;
	hodges[6] = 2.3287857323060801;
	hodges[7] = 2.3287857323060797;
	hodges[8] = 2.328785732306081;
	hodges[9] = 2.3287857323060801;
	hodges[10] = 2.3287857323060801;
	hodges[11] = 2.3287857323060797;
	hodges[12] = 2.3287857323060801;
	hodges[13] = 2.3287857323060801;
	hodges[14] = 2.3287857323060801;
	hodges[15] = 2.3287857323060788;
	hodges[16] = 2.3287857323060801;
	hodges[17] = 2.3287857323060801;
	hodges[18] = 2.3287857323060801;
	hodges[19] = 2.3287857323060788;
	hodges[20] = 2.328785732306081;
	hodges[21] = 2.3287857323060801;
	hodges[22] = 2.3287857323060801;
	hodges[23] = 2.3287857323060797;
	hodges[24] = 2.328785732306081;
	hodges[25] = 2.3287857323060801;
	hodges[26] = 2.3287857323060801;
	hodges[27] = 2.3287857323060797;
	hodges[28] = 2.3287857323060801;
	hodges[29] = 2.3287857323060801;
	hodges[30] = 2.3287857323060801;
	hodges[31] = 2.3287857323060788;
	hodges[32] = 2.3287857323060801;
	hodges[33] = 2.3287857323060801;
	hodges[34] = 2.3287857323060801;
	hodges[35] = 2.3287857323060788;
	hodges[36] = 2.328785732306081;
	hodges[37] = 2.3287857323060801;
	hodges[38] = 2.3287857323060801;
	hodges[39] = 2.3287857323060797;
	hodges[40] = 2.3287857323060801;
	hodges[41] = 2.3287857323060801;
	hodges[42] = 2.3287857323060801;
	hodges[43] = 2.3287857323060788;
	hodges[44] = 2.3287857323060801;
	hodges[45] = 2.3287857323060801;
	hodges[46] = 2.3287857323060801;
	hodges[47] = 2.3287857323060788;

	return 2.328785732306081;
}

#endif // MESH_H
