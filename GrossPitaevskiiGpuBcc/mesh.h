#define FACE_COUNT 4
#define VALUES_IN_BLOCK 12
#define INDICES_PER_BLOCK 48
const Vector3 BLOCK_WIDTH = Vector3(2.82843, 2.82843, 2.82843); // dimensions of unit block
const ddouble VOLUME = 1.88562; // volume of body elements
const bool IS_3D = true; // 3-dimensional
void getPositions(Buffer<Vector3> &pos)
{
	pos.resize(VALUES_IN_BLOCK);
	pos[0] = Vector3(0.353553, 1.06066, 2.47487);
	pos[1] = Vector3(2.47487, 0.353553, 1.06066);
	pos[2] = Vector3(2.47487, 1.06066, 0.353553);
	pos[3] = Vector3(2.47487, 1.06066, 1.76777);
	pos[4] = Vector3(2.47487, 1.76777, 1.06066);
	pos[5] = Vector3(0.353553, 2.47487, 1.06066);
	pos[6] = Vector3(1.06066, 2.47487, 0.353553);
	pos[7] = Vector3(1.06066, 2.47487, 1.76777);
	pos[8] = Vector3(1.76777, 2.47487, 1.06066);
	pos[9] = Vector3(1.06066, 0.353553, 2.47487);
	pos[10] = Vector3(1.06066, 1.76777, 2.47487);
	pos[11] = Vector3(1.76777, 1.06066, 2.47487);
}
ddouble getLaplacian(Buffer<int2> &ind, Buffer<ddouble> &hodges, const int nx, const int ny, const int nz) // nx, ny, nz in bytes
{
	ind.resize(INDICES_PER_BLOCK);
	ind[0] = make_int2(0, 9);
	ind[1] = make_int2(0, 10);
	ind[2] = make_int2(-nx + nz, 2);
	ind[3] = make_int2(-nx, 3);
	ind[4] = make_int2(0, 2);
	ind[5] = make_int2(0, 3);
	ind[6] = make_int2(nx - ny, 5);
	ind[7] = make_int2(-ny, 8);
	ind[8] = make_int2(0, 1);
	ind[9] = make_int2(0, 4);
	ind[10] = make_int2(nx - nz, 0);
	ind[11] = make_int2(-nz, 11);
	ind[12] = make_int2(0, 1);
	ind[13] = make_int2(0, 4);
	ind[14] = make_int2(nx, 0);
	ind[15] = make_int2(0, 11);
	ind[16] = make_int2(0, 2);
	ind[17] = make_int2(0, 3);
	ind[18] = make_int2(nx, 5);
	ind[19] = make_int2(0, 8);
	ind[20] = make_int2(0, 6);
	ind[21] = make_int2(0, 7);
	ind[22] = make_int2(-nx + ny, 1);
	ind[23] = make_int2(-nx, 4);
	ind[24] = make_int2(0, 5);
	ind[25] = make_int2(0, 8);
	ind[26] = make_int2(ny - nz, 9);
	ind[27] = make_int2(-nz, 10);
	ind[28] = make_int2(0, 5);
	ind[29] = make_int2(0, 8);
	ind[30] = make_int2(ny, 9);
	ind[31] = make_int2(0, 10);
	ind[32] = make_int2(0, 6);
	ind[33] = make_int2(0, 7);
	ind[34] = make_int2(ny, 1);
	ind[35] = make_int2(0, 4);
	ind[36] = make_int2(0, 0);
	ind[37] = make_int2(0, 11);
	ind[38] = make_int2(-ny + nz, 6);
	ind[39] = make_int2(-ny, 7);
	ind[40] = make_int2(0, 0);
	ind[41] = make_int2(0, 11);
	ind[42] = make_int2(nz, 6);
	ind[43] = make_int2(0, 7);
	ind[44] = make_int2(0, 9);
	ind[45] = make_int2(0, 10);
	ind[46] = make_int2(nz, 2);
	ind[47] = make_int2(0, 3);

	hodges.resize(INDICES_PER_BLOCK);
	hodges[0] = 1.5;
	hodges[1] = 1.5;
	hodges[2] = 1.5;
	hodges[3] = 1.5;
	hodges[4] = 1.5;
	hodges[5] = 1.5;
	hodges[6] = 1.5;
	hodges[7] = 1.5;
	hodges[8] = 1.5;
	hodges[9] = 1.5;
	hodges[10] = 1.5;
	hodges[11] = 1.5;
	hodges[12] = 1.5;
	hodges[13] = 1.5;
	hodges[14] = 1.5;
	hodges[15] = 1.5;
	hodges[16] = 1.5;
	hodges[17] = 1.5;
	hodges[18] = 1.5;
	hodges[19] = 1.5;
	hodges[20] = 1.5;
	hodges[21] = 1.5;
	hodges[22] = 1.5;
	hodges[23] = 1.5;
	hodges[24] = 1.5;
	hodges[25] = 1.5;
	hodges[26] = 1.5;
	hodges[27] = 1.5;
	hodges[28] = 1.5;
	hodges[29] = 1.5;
	hodges[30] = 1.5;
	hodges[31] = 1.5;
	hodges[32] = 1.5;
	hodges[33] = 1.5;
	hodges[34] = 1.5;
	hodges[35] = 1.5;
	hodges[36] = 1.5;
	hodges[37] = 1.5;
	hodges[38] = 1.5;
	hodges[39] = 1.5;
	hodges[40] = 1.5;
	hodges[41] = 1.5;
	hodges[42] = 1.5;
	hodges[43] = 1.5;
	hodges[44] = 1.5;
	hodges[45] = 1.5;
	hodges[46] = 1.5;
	hodges[47] = 1.5;

	return 1.5;
}
