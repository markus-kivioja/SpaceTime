#define FACE_COUNT 5
#define DUAL_EDGE_LENGTH 1.0868618817494704
#define VALUES_IN_BLOCK 12
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
ddouble getLaplacian(Buffer<int2> &ind, Buffer<ddouble> &hodges, const int nx, const int ny, const int nz, Buffer<int2> &indicesAndFaceCounts) // nx, ny, nz in bytes
{
	ind.resize(INDICES_PER_BLOCK);
	ind[0] = make_int2(0, 11);
	ind[1] = make_int2(0, 3);
	ind[2] = make_int2(0, 10);
	ind[3] = make_int2(0, 6);
	ind[4] = make_int2(-ny, 6);
	ind[5] = make_int2(0, 3);
	ind[6] = make_int2(-nz, 10);
	ind[7] = make_int2(-ny - nz, 11);
	ind[8] = make_int2(-nx - ny, 6);
	ind[9] = make_int2(0, 3);
	ind[10] = make_int2(-nx - nz, 10);
	ind[11] = make_int2(-ny - nz, 11);
	ind[12] = make_int2(0, 0);
	ind[13] = make_int2(0, 5);
	ind[14] = make_int2(0, 7);
	ind[15] = make_int2(0, 8);
	ind[16] = make_int2(0, 1);
	ind[17] = make_int2(0, 9);
	ind[18] = make_int2(0, 4);
	ind[19] = make_int2(0, 2);
	ind[20] = make_int2(-nx, 6);
	ind[21] = make_int2(0, 3);
	ind[22] = make_int2(-nz, 11);
	ind[23] = make_int2(-nx - nz, 10);
	ind[24] = make_int2(0, 6);
	ind[25] = make_int2(0, 3);
	ind[26] = make_int2(-nz, 11);
	ind[27] = make_int2(-nz, 10);
	ind[28] = make_int2(nx + ny, 2);
	ind[29] = make_int2(nx, 4);
	ind[30] = make_int2(nx + ny, 9);
	ind[31] = make_int2(ny, 1);
	ind[32] = make_int2(0, 5);
	ind[33] = make_int2(ny, 8);
	ind[34] = make_int2(nx, 7);
	ind[35] = make_int2(0, 0);
	ind[36] = make_int2(0, 11);
	ind[37] = make_int2(0, 3);
	ind[38] = make_int2(-nx, 10);
	ind[39] = make_int2(-nx, 6);
	ind[40] = make_int2(0, 10);
	ind[41] = make_int2(0, 3);
	ind[42] = make_int2(-ny, 11);
	ind[43] = make_int2(-ny, 6);
	ind[44] = make_int2(-nx, 10);
	ind[45] = make_int2(0, 3);
	ind[46] = make_int2(-ny, 11);
	ind[47] = make_int2(-nx - ny, 6);
	ind[48] = make_int2(nx + nz, 2);
	ind[49] = make_int2(nx, 9);
	ind[50] = make_int2(nx + nz, 4);
	ind[51] = make_int2(nz, 1);
	ind[52] = make_int2(0, 8);
	ind[53] = make_int2(nz, 5);
	ind[54] = make_int2(nx, 7);
	ind[55] = make_int2(0, 0);
	ind[56] = make_int2(ny + nz, 2);
	ind[57] = make_int2(ny, 9);
	ind[58] = make_int2(ny + nz, 1);
	ind[59] = make_int2(nz, 4);
	ind[60] = make_int2(0, 7);
	ind[61] = make_int2(nz, 5);
	ind[62] = make_int2(ny, 8);
	ind[63] = make_int2(0, 0);

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
	hodges[12] = 0.47618291773146471;
	hodges[13] = 0.47618291773146471;
	hodges[14] = 0.47618291773146471;
	hodges[15] = 0.47618291773146471;
	hodges[16] = 0.47618291773146471;
	hodges[17] = 0.47618291773146471;
	hodges[18] = 0.47618291773146471;
	hodges[19] = 0.47618291773146471;
	hodges[20] = 1.9047316709258588;
	hodges[21] = 1.9047316709258588;
	hodges[22] = 1.9047316709258588;
	hodges[23] = 1.9047316709258588;
	hodges[24] = 1.9047316709258588;
	hodges[25] = 1.9047316709258588;
	hodges[26] = 1.9047316709258588;
	hodges[27] = 1.9047316709258588;
	hodges[28] = 0.47618291773146471;
	hodges[29] = 0.47618291773146471;
	hodges[30] = 0.47618291773146471;
	hodges[31] = 0.47618291773146471;
	hodges[32] = 0.47618291773146471;
	hodges[33] = 0.47618291773146471;
	hodges[34] = 0.47618291773146471;
	hodges[35] = 0.47618291773146471;
	hodges[36] = 1.9047316709258588;
	hodges[37] = 1.9047316709258588;
	hodges[38] = 1.9047316709258588;
	hodges[39] = 1.9047316709258588;
	hodges[40] = 1.9047316709258588;
	hodges[41] = 1.9047316709258588;
	hodges[42] = 1.9047316709258588;
	hodges[43] = 1.9047316709258588;
	hodges[44] = 1.9047316709258588;
	hodges[45] = 1.9047316709258588;
	hodges[46] = 1.9047316709258588;
	hodges[47] = 1.9047316709258588;
	hodges[48] = 0.47618291773146471;
	hodges[49] = 0.47618291773146471;
	hodges[50] = 0.47618291773146471;
	hodges[51] = 0.47618291773146471;
	hodges[52] = 0.47618291773146471;
	hodges[53] = 0.47618291773146471;
	hodges[54] = 0.47618291773146471;
	hodges[55] = 0.47618291773146471;
	hodges[56] = 0.47618291773146471;
	hodges[57] = 0.47618291773146471;
	hodges[58] = 0.47618291773146471;
	hodges[59] = 0.47618291773146471;
	hodges[60] = 0.47618291773146471;
	hodges[61] = 0.47618291773146471;
	hodges[62] = 0.47618291773146471;
	hodges[63] = 0.47618291773146471;


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
