#define FACE_COUNT 4
#define DUAL_EDGE_LENGTH 2.0000000000000004
#define VALUES_IN_BLOCK 12
#define EDGES_IN_BLOCK 24
#define INDICES_PER_BLOCK 48
const Vector3 BLOCK_WIDTH = Vector3(5.6568542494923806, 5.6568542494923806, 5.6568542494923806); // dimensions of unit block
const ddouble VOLUME = 15.084944665313015; // volume of body elements
const bool IS_3D = true; // 3-dimensional
void getPositions(Buffer<Vector3> &pos)
{
	pos.resize(VALUES_IN_BLOCK);
	pos[0] = Vector3(0.70710678118654591, 2.1213203435596415, 4.9497474683058318);
	pos[1] = Vector3(4.9497474683058318, 0.70710678118654591, 2.1213203435596415);
	pos[2] = Vector3(4.9497474683058318, 2.1213203435596415, 0.70710678118654591);
	pos[3] = Vector3(4.9497474683058318, 2.1213203435596415, 3.5355339059327373);
	pos[4] = Vector3(4.9497474683058318, 3.5355339059327373, 2.1213203435596415);
	pos[5] = Vector3(0.70710678118654591, 4.9497474683058318, 2.1213203435596415);
	pos[6] = Vector3(2.1213203435596415, 4.9497474683058318, 0.70710678118654591);
	pos[7] = Vector3(2.1213203435596415, 4.9497474683058318, 3.5355339059327373);
	pos[8] = Vector3(3.5355339059327373, 4.9497474683058318, 2.1213203435596415);
	pos[9] = Vector3(2.1213203435596415, 0.70710678118654591, 4.9497474683058318);
	pos[10] = Vector3(2.1213203435596415, 3.5355339059327373, 4.9497474683058318);
	pos[11] = Vector3(3.5355339059327373, 2.1213203435596415, 4.9497474683058318);
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
	hodges[0] = 0.37499999999999989;
	hodges[1] = 0.37499999999999989;
	hodges[2] = 0.37500000000000011;
	hodges[3] = 0.37500000000000028;
	hodges[4] = 0.37499999999999989;
	hodges[5] = 0.37499999999999989;
	hodges[6] = 0.37500000000000011;
	hodges[7] = 0.37500000000000028;
	hodges[8] = -0.37500000000000011;
	hodges[9] = -0.37499999999999989;
	hodges[10] = 0.37499999999999989;
	hodges[11] = 0.37500000000000028;
	hodges[12] = -0.37500000000000028;
	hodges[13] = -0.37499999999999989;
	hodges[14] = 0.37499999999999983;
	hodges[15] = 0.37500000000000028;
	hodges[16] = -0.37499999999999989;
	hodges[17] = -0.37499999999999983;
	hodges[18] = 0.37500000000000006;
	hodges[19] = 0.37500000000000028;
	hodges[20] = -0.37500000000000011;
	hodges[21] = -0.37500000000000006;
	hodges[22] = 0.37499999999999989;
	hodges[23] = 0.37499999999999989;
	hodges[24] = -0.37499999999999989;
	hodges[25] = 0.37499999999999989;
	hodges[26] = 0.37500000000000011;
	hodges[27] = 0.37500000000000028;
	hodges[28] = -0.37499999999999989;
	hodges[29] = 0.37499999999999983;
	hodges[30] = 0.37500000000000006;
	hodges[31] = 0.37500000000000028;
	hodges[32] = -0.37500000000000028;
	hodges[33] = -0.37500000000000028;
	hodges[34] = -0.37499999999999989;
	hodges[35] = -0.37499999999999983;
	hodges[36] = -0.37499999999999989;
	hodges[37] = -0.37500000000000011;
	hodges[38] = -0.37500000000000006;
	hodges[39] = 0.37499999999999989;
	hodges[40] = -0.37499999999999989;
	hodges[41] = -0.37500000000000028;
	hodges[42] = -0.37500000000000028;
	hodges[43] = 0.37499999999999983;
	hodges[44] = -0.37500000000000028;
	hodges[45] = -0.37500000000000028;
	hodges[46] = -0.37499999999999989;
	hodges[47] = -0.37499999999999983;

	return 0.37500000000000028;
}
