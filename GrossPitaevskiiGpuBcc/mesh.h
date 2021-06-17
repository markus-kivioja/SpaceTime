#define FACE_COUNT 4
#define VALUES_IN_BLOCK 12
#define INDICES_PER_BLOCK 48
const Vector3 BLOCK_WIDTH = Vector3(1.84, 1.84, 1.84); // dimensions of unit block
const ddouble VOLUME = 0.519125; // volume of body elements
const bool IS_3D = true; // 3-dimensional
void getPositions(Buffer<Vector3> &pos)
{
	pos.resize(VALUES_IN_BLOCK);
	pos[0] = Vector3(0.23, 0.69, 1.61);
	pos[1] = Vector3(1.61, 0.23, 0.69);
	pos[2] = Vector3(1.61, 0.69, 0.23);
	pos[3] = Vector3(1.61, 0.69, 1.15);
	pos[4] = Vector3(1.61, 1.15, 0.69);
	pos[5] = Vector3(0.23, 1.61, 0.69);
	pos[6] = Vector3(0.69, 1.61, 0.23);
	pos[7] = Vector3(0.69, 1.61, 1.15);
	pos[8] = Vector3(1.15, 1.61, 0.69);
	pos[9] = Vector3(0.69, 0.23, 1.61);
	pos[10] = Vector3(0.69, 1.15, 1.61);
	pos[11] = Vector3(1.15, 0.69, 1.61);
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
	hodges[0] = 3.54442;
	hodges[1] = 3.54442;
	hodges[2] = 3.54442;
	hodges[3] = 3.54442;
	hodges[4] = 3.54442;
	hodges[5] = 3.54442;
	hodges[6] = 3.54442;
	hodges[7] = 3.54442;
	hodges[8] = 3.54442;
	hodges[9] = 3.54442;
	hodges[10] = 3.54442;
	hodges[11] = 3.54442;
	hodges[12] = 3.54442;
	hodges[13] = 3.54442;
	hodges[14] = 3.54442;
	hodges[15] = 3.54442;
	hodges[16] = 3.54442;
	hodges[17] = 3.54442;
	hodges[18] = 3.54442;
	hodges[19] = 3.54442;
	hodges[20] = 3.54442;
	hodges[21] = 3.54442;
	hodges[22] = 3.54442;
	hodges[23] = 3.54442;
	hodges[24] = 3.54442;
	hodges[25] = 3.54442;
	hodges[26] = 3.54442;
	hodges[27] = 3.54442;
	hodges[28] = 3.54442;
	hodges[29] = 3.54442;
	hodges[30] = 3.54442;
	hodges[31] = 3.54442;
	hodges[32] = 3.54442;
	hodges[33] = 3.54442;
	hodges[34] = 3.54442;
	hodges[35] = 3.54442;
	hodges[36] = 3.54442;
	hodges[37] = 3.54442;
	hodges[38] = 3.54442;
	hodges[39] = 3.54442;
	hodges[40] = 3.54442;
	hodges[41] = 3.54442;
	hodges[42] = 3.54442;
	hodges[43] = 3.54442;
	hodges[44] = 3.54442;
	hodges[45] = 3.54442;
	hodges[46] = 3.54442;
	hodges[47] = 3.54442;

	return 3.54442;
}
