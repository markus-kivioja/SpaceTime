#define FACE_COUNT 4
#define DUAL_EDGE_LENGTH 1.0000000000000002
#define VALUES_IN_BLOCK 12
#define EDGES_IN_BLOCK 24
#define INDICES_PER_BLOCK 48
const Vector3 BLOCK_WIDTH = Vector3(2.8284271247461903, 2.8284271247461903, 2.8284271247461903); // dimensions of unit block
const ddouble VOLUME = 1.8856180831641269; // volume of body elements
const bool IS_3D = true; // 3-dimensional
void getPositions(Buffer<Vector3> &pos)
{
	pos.resize(VALUES_IN_BLOCK);
	pos[0] = Vector3(0.35355339059327295, 1.0606601717798207, 2.4748737341529159);
	pos[1] = Vector3(2.4748737341529159, 0.35355339059327295, 1.0606601717798207);
	pos[2] = Vector3(2.4748737341529159, 1.0606601717798207, 0.35355339059327295);
	pos[3] = Vector3(2.4748737341529159, 1.0606601717798207, 1.7677669529663687);
	pos[4] = Vector3(2.4748737341529159, 1.7677669529663687, 1.0606601717798207);
	pos[5] = Vector3(0.35355339059327295, 2.4748737341529159, 1.0606601717798207);
	pos[6] = Vector3(1.0606601717798207, 2.4748737341529159, 0.35355339059327295);
	pos[7] = Vector3(1.0606601717798207, 2.4748737341529159, 1.7677669529663687);
	pos[8] = Vector3(1.7677669529663687, 2.4748737341529159, 1.0606601717798207);
	pos[9] = Vector3(1.0606601717798207, 0.35355339059327295, 2.4748737341529159);
	pos[10] = Vector3(1.0606601717798207, 1.7677669529663687, 2.4748737341529159);
	pos[11] = Vector3(1.7677669529663687, 1.0606601717798207, 2.4748737341529159);
}
ddouble getLaplacian(Buffer<ddouble>& hodges, Buffer<int3>& d0, Buffer<int2>& d1, const int d0x, const int d0y, const int d0z, const int d1x, const int d1y, const int d1z) // offsets in bytes
{
	d0.resize(EDGES_IN_BLOCK);
	d0[0] = {make_int3(0, 0, 9)};
	d0[1] = {make_int3(0, 0, 10)};
	d0[2] = {make_int3(0, -d0x + d0z, 2)};
	d0[3] = {make_int3(0, -d0x, 3)};
	d0[4] = {make_int3(1, 0, 2)};
	d0[5] = {make_int3(1, 0, 3)};
	d0[6] = {make_int3(1, d0x - d0y, 5)};
	d0[7] = {make_int3(1, -d0y, 8)};
	d0[8] = {make_int3(2, 0, 4)};
	d0[9] = {make_int3(2, -d0z, 11)};
	d0[10] = {make_int3(3, 0, 4)};
	d0[11] = {make_int3(3, 0, 11)};
	d0[12] = {make_int3(4, d0x, 5)};
	d0[13] = {make_int3(4, 0, 8)};
	d0[14] = {make_int3(5, 0, 6)};
	d0[15] = {make_int3(5, 0, 7)};
	d0[16] = {make_int3(6, 0, 8)};
	d0[17] = {make_int3(6, d0y - d0z, 9)};
	d0[18] = {make_int3(6, -d0z, 10)};
	d0[19] = {make_int3(7, 0, 8)};
	d0[20] = {make_int3(7, d0y, 9)};
	d0[21] = {make_int3(7, 0, 10)};
	d0[22] = {make_int3(9, 0, 11)};
	d0[23] = {make_int3(10, 0, 11)};

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
	d1[8] = make_int2(-(-d1x + d1z), 2);
	d1[9] = make_int2(-(0), 4);
	d1[10] = make_int2(0, 8);
	d1[11] = make_int2(0, 9);
	//3
	d1[12] = make_int2(-(-d1x), 3);
	d1[13] = make_int2(-(0), 5);
	d1[14] = make_int2(0, 10);
	d1[15] = make_int2(0, 11);
	//4
	d1[16] = make_int2(-(0), 8);
	d1[17] = make_int2(-(0), 10);
	d1[18] = make_int2(0, 12);
	d1[19] = make_int2(0, 13);
	//5
	d1[20] = make_int2(-(d1x - d1y), 6);
	d1[21] = make_int2(-(d1x), 12);
	d1[22] = make_int2(0, 14);
	d1[23] = make_int2(0, 15);
	//6
	d1[24] = make_int2(-(0), 14);
	d1[25] = make_int2(0, 16);
	d1[26] = make_int2(0, 17);
	d1[27] = make_int2(0, 18);
	//7
	d1[28] = make_int2(-(0), 15);
	d1[29] = make_int2(0, 19);
	d1[30] = make_int2(0, 20);
	d1[31] = make_int2(0, 21);
	//8
	d1[32] = make_int2(-(-d1y), 7);
	d1[33] = make_int2(-(0), 13);
	d1[34] = make_int2(-(0), 16);
	d1[35] = make_int2(-(0), 19);
	//9
	d1[36] = make_int2(-(0), 0);
	d1[37] = make_int2(-(d1y - d1z), 17);
	d1[38] = make_int2(-(d1y), 20);
	d1[39] = make_int2(0, 22);
	//10
	d1[40] = make_int2(-(0), 1);
	d1[41] = make_int2(-(-d1z), 18);
	d1[42] = make_int2(-(0), 21);
	d1[43] = make_int2(0, 23);
	//11
	d1[44] = make_int2(-(-d1z), 9);
	d1[45] = make_int2(-(0), 11);
	d1[46] = make_int2(-(0), 22);
	d1[47] = make_int2(-(0), 23);

	hodges.resize(INDICES_PER_BLOCK);
	hodges[0] = 1.4999999999999996;
	hodges[1] = 1.4999999999999996;
	hodges[2] = 1.5000000000000004;
	hodges[3] = 1.5000000000000011;
	hodges[4] = 1.4999999999999996;
	hodges[5] = 1.4999999999999996;
	hodges[6] = 1.5000000000000004;
	hodges[7] = 1.5000000000000011;
	hodges[8] = -1.5000000000000004;
	hodges[9] = -1.4999999999999996;
	hodges[10] = 1.4999999999999996;
	hodges[11] = 1.5000000000000011;
	hodges[12] = -1.5000000000000011;
	hodges[13] = -1.4999999999999996;
	hodges[14] = 1.4999999999999993;
	hodges[15] = 1.5000000000000011;
	hodges[16] = -1.4999999999999996;
	hodges[17] = -1.4999999999999993;
	hodges[18] = 1.5000000000000002;
	hodges[19] = 1.5000000000000011;
	hodges[20] = -1.5000000000000004;
	hodges[21] = -1.5000000000000002;
	hodges[22] = 1.4999999999999996;
	hodges[23] = 1.4999999999999996;
	hodges[24] = -1.4999999999999996;
	hodges[25] = 1.4999999999999996;
	hodges[26] = 1.5000000000000004;
	hodges[27] = 1.5000000000000011;
	hodges[28] = -1.4999999999999996;
	hodges[29] = 1.4999999999999993;
	hodges[30] = 1.5000000000000002;
	hodges[31] = 1.5000000000000011;
	hodges[32] = -1.5000000000000011;
	hodges[33] = -1.5000000000000011;
	hodges[34] = -1.4999999999999996;
	hodges[35] = -1.4999999999999993;
	hodges[36] = -1.4999999999999996;
	hodges[37] = -1.5000000000000004;
	hodges[38] = -1.5000000000000002;
	hodges[39] = 1.4999999999999996;
	hodges[40] = -1.4999999999999996;
	hodges[41] = -1.5000000000000011;
	hodges[42] = -1.5000000000000011;
	hodges[43] = 1.4999999999999993;
	hodges[44] = -1.5000000000000011;
	hodges[45] = -1.5000000000000011;
	hodges[46] = -1.4999999999999996;
	hodges[47] = -1.4999999999999993;

	return 1.5000000000000011;
}
