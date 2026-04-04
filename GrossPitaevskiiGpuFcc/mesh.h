#define FACE_COUNT 5
#define DUAL_EDGE_LENGTH 1.0868618817494704
#define VALUES_IN_BLOCK 12
#define EDGES_IN_BLOCK 32
#define INDICES_PER_BLOCK 64
const Vector3 BLOCK_WIDTH = Vector3(2.5099999999999998, 2.5099999999999998, 2.5099999999999998); // dimensions of unit block
const ddouble VOLUME = 0.65888545833333312; // volume of body elements
const bool IS_3D = true; // 3-dimensional
void getPositions(Buffer<Vector3> &pos)
{
	pos.resize(VALUES_IN_BLOCK);
	pos[0] = Vector3(1.5687500000000001, 1.5687500000000001, 1.5687500000000001);
	pos[1] = Vector3(1.5687500000000001, 0.3137500000000002, 0.3137500000000002);
	pos[2] = Vector3(0.3137500000000002, 0.3137500000000002, 0.3137500000000002);
	pos[3] = Vector3(0.94125000000000014, 0.94125000000000014, 0.94125000000000014);
	pos[4] = Vector3(0.3137500000000002, 1.5687500000000001, 0.3137500000000002);
	pos[5] = Vector3(1.5687500000000001, 1.5687500000000001, 0.3137500000000002);
	pos[6] = Vector3(2.19625, 2.19625, 0.94125000000000014);
	pos[7] = Vector3(0.3137500000000002, 1.5687500000000001, 1.5687500000000001);
	pos[8] = Vector3(1.5687500000000001, 0.3137500000000002, 1.5687500000000001);
	pos[9] = Vector3(0.3137500000000002, 0.3137500000000002, 1.5687500000000001);
	pos[10] = Vector3(2.19625, 0.94125000000000014, 2.19625);
	pos[11] = Vector3(0.94125000000000014, 2.19625, 2.19625);
}
ddouble getLaplacian(Buffer<int2> &bodyIndices, Buffer<ddouble> &hodges, const int nx, const int ny, const int nz, Buffer<int2> &indicesAndFaceCounts) // nx, ny, nz in bytes
{
	d0.resize(EDGES_IN_BLOCK);
	d0[0] = {make_int3(0, 0, 11)};
	d0[1] = {make_int3(0, 0, 3)};
	d0[2] = {make_int3(0, 0, 10)};
	d0[3] = {make_int3(0, 0, 6)};
	d0[4] = {make_int3(1, -d0y, 6)};
	d0[5] = {make_int3(1, 0, 3)};
	d0[6] = {make_int3(1, -d0z, 10)};
	d0[7] = {make_int3(1, -d0y - d0z, 11)};
	d0[8] = {make_int3(2, -d0x - d0y, 6)};
	d0[9] = {make_int3(2, 0, 3)};
	d0[10] = {make_int3(2, -d0x - d0z, 10)};
	d0[11] = {make_int3(2, -d0y - d0z, 11)};
	d0[12] = {make_int3(3, 0, 5)};
	d0[13] = {make_int3(3, 0, 7)};
	d0[14] = {make_int3(3, 0, 8)};
	d0[15] = {make_int3(3, 0, 9)};
	d0[16] = {make_int3(3, 0, 4)};
	d0[17] = {make_int3(4, -d0x, 6)};
	d0[18] = {make_int3(4, -d0z, 11)};
	d0[19] = {make_int3(4, -d0x - d0z, 10)};
	d0[20] = {make_int3(5, 0, 6)};
	d0[21] = {make_int3(5, -d0z, 11)};
	d0[22] = {make_int3(5, -d0z, 10)};
	d0[23] = {make_int3(6, d0x + d0y, 9)};
	d0[24] = {make_int3(6, d0y, 8)};
	d0[25] = {make_int3(6, d0x, 7)};
	d0[26] = {make_int3(7, 0, 11)};
	d0[27] = {make_int3(7, -d0x, 10)};
	d0[28] = {make_int3(8, 0, 10)};
	d0[29] = {make_int3(8, -d0y, 11)};
	d0[30] = {make_int3(9, -d0x, 10)};
	d0[31] = {make_int3(9, -d0y, 11)};

	d1.resize(INDICES_PER_BLOCK);
	//0
	d1[0] = make_int2(0, 0);
	d1[1] = make_int2(0, 1);
	d1[2] = make_int2(0, 2);
	d1[3] = make_int2(0, 3);
	//1
	d1[4] = make_int2(0, 4);
	d1[5] = make_int2(0, 5);
	d1[6] = make_int2(0, 6);
	d1[7] = make_int2(0, 7);
	//2
	d1[8] = make_int2(0, 8);
	d1[9] = make_int2(0, 9);
	d1[10] = make_int2(0, 10);
	d1[11] = make_int2(0, 11);
	//3
	d1[12] = make_int2(-(0), 1);
	d1[13] = make_int2(-(0), 5);
	d1[14] = make_int2(-(0), 9);
	d1[15] = make_int2(0, 12);
	d1[16] = make_int2(0, 13);
	d1[17] = make_int2(0, 14);
	d1[18] = make_int2(0, 15);
	d1[19] = make_int2(0, 16);
	//4
	d1[20] = make_int2(-(0), 16);
	d1[21] = make_int2(0, 17);
	d1[22] = make_int2(0, 18);
	d1[23] = make_int2(0, 19);
	//5
	d1[24] = make_int2(-(0), 12);
	d1[25] = make_int2(0, 20);
	d1[26] = make_int2(0, 21);
	d1[27] = make_int2(0, 22);
	//6
	d1[28] = make_int2(-(0), 3);
	d1[29] = make_int2(-(-d1y), 4);
	d1[30] = make_int2(-(-d1x - d1y), 8);
	d1[31] = make_int2(-(-d1x), 17);
	d1[32] = make_int2(-(0), 20);
	d1[33] = make_int2(0, 23);
	d1[34] = make_int2(0, 24);
	d1[35] = make_int2(0, 25);
	//7
	d1[36] = make_int2(-(0), 13);
	d1[37] = make_int2(-(d1x), 25);
	d1[38] = make_int2(0, 26);
	d1[39] = make_int2(0, 27);
	//8
	d1[40] = make_int2(-(0), 14);
	d1[41] = make_int2(-(d1y), 24);
	d1[42] = make_int2(0, 28);
	d1[43] = make_int2(0, 29);
	//9
	d1[44] = make_int2(-(0), 15);
	d1[45] = make_int2(-(d1x + d1y), 23);
	d1[46] = make_int2(0, 30);
	d1[47] = make_int2(0, 31);
	//10
	d1[48] = make_int2(-(0), 2);
	d1[49] = make_int2(-(-d1z), 6);
	d1[50] = make_int2(-(-d1x - d1z), 10);
	d1[51] = make_int2(-(-d1x - d1z), 19);
	d1[52] = make_int2(-(-d1z), 22);
	d1[53] = make_int2(-(-d1x), 27);
	d1[54] = make_int2(-(0), 28);
	d1[55] = make_int2(-(-d1x), 30);
	//11
	d1[56] = make_int2(-(0), 0);
	d1[57] = make_int2(-(-d1y - d1z), 7);
	d1[58] = make_int2(-(-d1y - d1z), 11);
	d1[59] = make_int2(-(-d1z), 18);
	d1[60] = make_int2(-(-d1z), 21);
	d1[61] = make_int2(-(0), 26);
	d1[62] = make_int2(-(-d1y), 29);
	d1[63] = make_int2(-(-d1y), 31);

	hodges.resize(INDICES_PER_BLOCK);
	hodges[0] = 1.9047316709258588;
	hodges[1] = 1.9047316709258588;
	hodges[2] = 1.9047316709258588;
	hodges[3] = 1.9047316709258588;
	hodges[4] = 1.9047316709258588;
	hodges[5] = 1.9047316709258588;
	hodges[6] = 1.9047316709258588;
	hodges[7] = 1.9047316709258588;
	hodges[8] = 1.9047316709258588;
	hodges[9] = 1.9047316709258588;
	hodges[10] = 1.9047316709258588;
	hodges[11] = 1.9047316709258588;
	hodges[12] = -1.9047316709258588;
	hodges[13] = -1.9047316709258588;
	hodges[14] = -1.9047316709258588;
	hodges[15] = 0.47618291773146471;
	hodges[16] = 0.47618291773146471;
	hodges[17] = 0.47618291773146471;
	hodges[18] = 0.47618291773146471;
	hodges[19] = 0.47618291773146471;
	hodges[20] = -0.47618291773146471;
	hodges[21] = 1.9047316709258588;
	hodges[22] = 1.9047316709258588;
	hodges[23] = 1.9047316709258588;
	hodges[24] = -0.47618291773146471;
	hodges[25] = 1.9047316709258588;
	hodges[26] = 1.9047316709258588;
	hodges[27] = 1.9047316709258588;
	hodges[28] = -1.9047316709258588;
	hodges[29] = -1.9047316709258588;
	hodges[30] = -1.9047316709258588;
	hodges[31] = -1.9047316709258588;
	hodges[32] = -1.9047316709258588;
	hodges[33] = 0.47618291773146471;
	hodges[34] = 0.47618291773146471;
	hodges[35] = 0.47618291773146471;
	hodges[36] = -0.47618291773146471;
	hodges[37] = -0.47618291773146471;
	hodges[38] = 1.9047316709258588;
	hodges[39] = 1.9047316709258588;
	hodges[40] = -0.47618291773146471;
	hodges[41] = -0.47618291773146471;
	hodges[42] = 1.9047316709258588;
	hodges[43] = 1.9047316709258588;
	hodges[44] = -0.47618291773146471;
	hodges[45] = -0.47618291773146471;
	hodges[46] = 1.9047316709258588;
	hodges[47] = 1.9047316709258588;
	hodges[48] = -1.9047316709258588;
	hodges[49] = -1.9047316709258588;
	hodges[50] = -1.9047316709258588;
	hodges[51] = -1.9047316709258588;
	hodges[52] = -1.9047316709258588;
	hodges[53] = -1.9047316709258588;
	hodges[54] = -1.9047316709258588;
	hodges[55] = -1.9047316709258588;
	hodges[56] = -1.9047316709258588;
	hodges[57] = -1.9047316709258588;
	hodges[58] = -1.9047316709258588;
	hodges[59] = -1.9047316709258588;
	hodges[60] = -1.9047316709258588;
	hodges[61] = -1.9047316709258588;
	hodges[62] = -1.9047316709258588;
	hodges[63] = -1.9047316709258588;


	indicesAndFaceCounts.resize(VALUES_IN_BLOCK);
	indicesAndFaceCounts[0] = make_int2(0, 4);
	indicesAndFaceCounts[1] = make_int2(4, 4);
	indicesAndFaceCounts[2] = make_int2(8, 4);
	indicesAndFaceCounts[3] = make_int2(12, 8);
	indicesAndFaceCounts[4] = make_int2(20, 4);
	indicesAndFaceCounts[5] = make_int2(24, 4);
	indicesAndFaceCounts[6] = make_int2(28, 8);
	indicesAndFaceCounts[7] = make_int2(36, 4);
	indicesAndFaceCounts[8] = make_int2(40, 4);
	indicesAndFaceCounts[9] = make_int2(44, 4);
	indicesAndFaceCounts[10] = make_int2(48, 8);
	indicesAndFaceCounts[11] = make_int2(56, 8);

	return 1.9047316709258588;
}
