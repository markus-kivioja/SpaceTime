#define FACE_COUNT 4
#define DUAL_EDGE_LENGTH 0.5740991584648073
#define VALUES_IN_BLOCK 136
#define INDICES_PER_BLOCK 544
const Vector3 BLOCK_WIDTH = Vector3(4.5, 4.5, 4.5); // dimensions of unit block
const ddouble VOLUME = 0.7119140625; // volume of body elements
const bool IS_3D = true; // 3-dimensional
void getPositions(Buffer<Vector3> &pos)
{
	pos.resize(VALUES_IN_BLOCK);
	pos[0] = Vector3(1.359375, 2.90625, 2.015625);
	pos[1] = Vector3(4.03125, 2.484375, 2.484375);
	pos[2] = Vector3(1.359375, 4.265625, 0.65625000000000011);
	pos[3] = Vector3(1.59375, 2.484375, 3.140625);
	pos[4] = Vector3(4.265625, 2.015625, 2.71875);
	pos[5] = Vector3(3.84375, 3.140625, 0.234375);
	pos[6] = Vector3(2.015625, 0.46875, 2.015625);
	pos[7] = Vector3(2.484375, 4.03125, 2.484375);
	pos[8] = Vector3(1.6875, 3.9375, 3.9375);
	pos[9] = Vector3(2.484375, 3.140625, 1.59375);
	pos[10] = Vector3(2.015625, 2.015625, 0.46875);
	pos[11] = Vector3(4.03125, 3.140625, 3.140625);
	pos[12] = Vector3(0.056249999999999994, 0.056249999999999967, 3.3187500000000001);
	pos[13] = Vector3(2.8125, 2.8125, 2.8125);
	pos[14] = Vector3(1.0687500000000001, 2.3062499999999999, 0.056249999999999967);
	pos[15] = Vector3(1.6875, 1.6875, 1.6875);
	pos[16] = Vector3(3.3187500000000001, 2.3062499999999999, 2.3062499999999999);
	pos[17] = Vector3(1.359375, 0.46875, 1.359375);
	pos[18] = Vector3(1.359375, 1.359375, 0.46875);
	pos[19] = Vector3(1.0687500000000001, 3.3187500000000001, 1.0687500000000001);
	pos[20] = Vector3(1.359375, 2.015625, 2.90625);
	pos[21] = Vector3(0.056249999999999967, 3.3187500000000001, 0.056249999999999967);
	pos[22] = Vector3(0.890625, 1.59375, 0.234375);
	pos[23] = Vector3(0.056249999999999967, 2.3062499999999999, 1.0687500000000001);
	pos[24] = Vector3(1.59375, 0.234375, 0.890625);
	pos[25] = Vector3(0.056249999999999994, 1.0687500000000001, 2.3062499999999999);
	pos[26] = Vector3(2.3062499999999999, 1.0687500000000001, 0.056250000000000022);
	pos[27] = Vector3(1.1812499999999999, 2.1937500000000001, 2.1937500000000001);
	pos[28] = Vector3(3.3187500000000001, 0.056249999999999967, 0.056250000000000022);
	pos[29] = Vector3(2.015625, 2.71875, 4.265625);
	pos[30] = Vector3(0.5625, 0.5625, 2.8125);
	pos[31] = Vector3(0.234375, 1.78125, 2.484375);
	pos[32] = Vector3(3.3187500000000001, 3.3187500000000001, 3.3187500000000001);
	pos[33] = Vector3(0.234375, 0.890625, 1.59375);
	pos[34] = Vector3(0.46875, 3.609375, 3.609375);
	pos[35] = Vector3(3.609375, 2.71875, 1.359375);
	pos[36] = Vector3(2.3062499999999999, 2.3062499999999999, 3.3187500000000001);
	pos[37] = Vector3(1.0687500000000001, 0.056249999999999994, 2.3062499999999999);
	pos[38] = Vector3(0.890625, 0.234375, 1.59375);
	pos[39] = Vector3(1.78125, 0.234375, 2.484375);
	pos[40] = Vector3(0.5625, 2.8125, 0.5625);
	pos[41] = Vector3(0.46875, 2.015625, 2.015625);
	pos[42] = Vector3(1.78125, 2.484375, 0.234375);
	pos[43] = Vector3(0.234375, 2.484375, 1.78125);
	pos[44] = Vector3(4.265625, 2.71875, 2.015625);
	pos[45] = Vector3(3.84375, 0.890625, 2.484375);
	pos[46] = Vector3(0.46875, 1.359375, 1.359375);
	pos[47] = Vector3(0.234375, 1.59375, 0.890625);
	pos[48] = Vector3(2.3062499999999999, 0.056249999999999967, 1.0687500000000001);
	pos[49] = Vector3(1.1812499999999999, 1.1812499999999999, 1.1812499999999999);
	pos[50] = Vector3(3.3187500000000001, 1.0687500000000001, 1.0687500000000001);
	pos[51] = Vector3(2.484375, 0.234375, 1.78125);
	pos[52] = Vector3(2.015625, 1.359375, 2.90625);
	pos[53] = Vector3(0.890625, 3.140625, 1.78125);
	pos[54] = Vector3(1.59375, 0.890625, 0.234375);
	pos[55] = Vector3(2.8125, 0.5625, 0.5625);
	pos[56] = Vector3(3.609375, 0.65625, 2.015625);
	pos[57] = Vector3(2.484375, 1.78125, 0.234375);
	pos[58] = Vector3(2.1937500000000001, 2.1937500000000001, 1.1812499999999999);
	pos[59] = Vector3(2.1937500000000001, 1.1812499999999999, 2.1937500000000001);
	pos[60] = Vector3(0.65625, 3.609375, 2.015625);
	pos[61] = Vector3(1.59375, 3.140625, 2.484375);
	pos[62] = Vector3(3.140625, 2.484375, 1.59375);
	pos[63] = Vector3(2.90625, 2.015625, 1.359375);
	pos[64] = Vector3(3.140625, 1.78125, 0.890625);
	pos[65] = Vector3(3.140625, 1.59375, 2.484375);
	pos[66] = Vector3(2.90625, 1.359375, 2.015625);
	pos[67] = Vector3(3.140625, 0.890625, 1.78125);
	pos[68] = Vector3(3.140625, 3.84375, 0.234375);
	pos[69] = Vector3(1.1812499999999999, 4.4437499999999996, 4.4437499999999996);
	pos[70] = Vector3(3.609375, 1.359375, 2.71875);
	pos[71] = Vector3(4.03125, 0.890625, 0.890625);
	pos[72] = Vector3(4.03125, 0.234375, 0.234375);
	pos[73] = Vector3(4.265625, 0.65625000000000011, 1.359375);
	pos[74] = Vector3(3.84375, 0.234375, 3.140625);
	pos[75] = Vector3(2.3062499999999999, 3.3187500000000001, 2.3062499999999999);
	pos[76] = Vector3(2.015625, 3.609375, 0.65625);
	pos[77] = Vector3(2.1937500000000001, 4.4437499999999996, 3.4312499999999999);
	pos[78] = Vector3(4.265625, 1.359375, 0.65625000000000011);
	pos[79] = Vector3(3.84375, 2.484375, 0.890625);
	pos[80] = Vector3(3.609375, 2.015625, 0.65625);
	pos[81] = Vector3(2.71875, 3.609375, 1.359375);
	pos[82] = Vector3(1.78125, 3.140625, 0.890625);
	pos[83] = Vector3(2.015625, 2.90625, 1.359375);
	pos[84] = Vector3(3.140625, 4.03125, 3.140625);
	pos[85] = Vector3(0.890625, 0.890625, 4.03125);
	pos[86] = Vector3(1.359375, 3.609375, 2.71875);
	pos[87] = Vector3(0.890625, 4.03125, 0.890625);
	pos[88] = Vector3(0.234375, 4.03125, 0.234375);
	pos[89] = Vector3(4.4437499999999996, 3.4312499999999999, 2.1937500000000001);
	pos[90] = Vector3(2.015625, 4.265625, 2.71875);
	pos[91] = Vector3(0.890625, 3.84375, 2.484375);
	pos[92] = Vector3(0.65625000000000011, 4.265625, 1.359375);
	pos[93] = Vector3(0.234375, 3.84375, 3.140625);
	pos[94] = Vector3(2.90625, 4.265625, 3.609375);
	pos[95] = Vector3(3.4312499999999999, 3.4312499999999999, 1.1812499999999999);
	pos[96] = Vector3(2.71875, 4.265625, 2.015625);
	pos[97] = Vector3(2.484375, 3.84375, 0.890625);
	pos[98] = Vector3(3.609375, 3.609375, 0.46875);
	pos[99] = Vector3(4.265625, 4.265625, 0.46875000000000006);
	pos[100] = Vector3(3.9375, 3.9375, 1.6875);
	pos[101] = Vector3(4.4437499999999996, 4.4437499999999996, 1.1812499999999999);
	pos[102] = Vector3(3.4312499999999999, 4.4437499999999996, 2.1937500000000001);
	pos[103] = Vector3(3.609375, 4.265625, 2.90625);
	pos[104] = Vector3(4.265625, 3.609375, 2.90625);
	pos[105] = Vector3(2.015625, 0.65625, 3.609375);
	pos[106] = Vector3(2.484375, 1.59375, 3.140625);
	pos[107] = Vector3(0.234375, 3.140625, 3.84375);
	pos[108] = Vector3(1.78125, 0.890625, 3.140625);
	pos[109] = Vector3(0.890625, 1.78125, 3.140625);
	pos[110] = Vector3(2.484375, 2.484375, 4.03125);
	pos[111] = Vector3(3.140625, 3.140625, 4.03125);
	pos[112] = Vector3(1.0687500000000001, 1.0687500000000001, 3.3187500000000001);
	pos[113] = Vector3(1.359375, 0.65625000000000011, 4.265625);
	pos[114] = Vector3(1.1812499999999999, 3.4312499999999999, 3.4312499999999999);
	pos[115] = Vector3(0.234375, 0.234375, 4.03125);
	pos[116] = Vector3(1.359375, 2.71875, 3.609375);
	pos[117] = Vector3(0.65625000000000011, 1.359375, 4.265625);
	pos[118] = Vector3(0.890625, 2.484375, 3.84375);
	pos[119] = Vector3(0.65625, 2.015625, 3.609375);
	pos[120] = Vector3(3.140625, 0.234375, 3.84375);
	pos[121] = Vector3(3.609375, 0.46875, 3.609375);
	pos[122] = Vector3(3.4312499999999999, 1.1812499999999999, 3.4312499999999999);
	pos[123] = Vector3(2.484375, 0.890625, 3.84375);
	pos[124] = Vector3(2.71875, 1.359375, 3.609375);
	pos[125] = Vector3(2.71875, 2.015625, 4.265625);
	pos[126] = Vector3(4.265625, 2.90625, 3.609375);
	pos[127] = Vector3(3.4312499999999999, 2.1937500000000001, 4.4437499999999996);
	pos[128] = Vector3(4.4437499999999996, 1.1812499999999999, 4.4437499999999996);
	pos[129] = Vector3(4.265625, 0.46875000000000006, 4.265625);
	pos[130] = Vector3(4.4437499999999996, 2.1937500000000001, 3.4312499999999999);
	pos[131] = Vector3(3.9375, 1.6875, 3.9375);
	pos[132] = Vector3(3.609375, 2.90625, 4.265625);
	pos[133] = Vector3(2.1937500000000001, 3.4312499999999999, 4.4437499999999996);
	pos[134] = Vector3(0.46875000000000006, 4.265625, 4.265625);
	pos[135] = Vector3(2.90625, 3.609375, 4.265625);
}
ddouble getLaplacian(Buffer<int3> &blockDirs, Buffer<int> &valueInds, Buffer<ddouble> &hodges)
{
	blockDirs.resize(INDICES_PER_BLOCK);
	// Dual node idx:  0
	blockDirs[0] = make_int3(0, 0, 0);
	blockDirs[1] = make_int3(0, 0, 0);
	blockDirs[2] = make_int3(0, 0, 0);
	blockDirs[3] = make_int3(0, 0, 0);
	// Dual node idx:  1
	blockDirs[4] = make_int3(0, 0, 0);
	blockDirs[5] = make_int3(0, 0, 0);
	blockDirs[6] = make_int3(0, 0, 0);
	blockDirs[7] = make_int3(0, 0, 0);
	// Dual node idx:  2
	blockDirs[8] = make_int3(0, 0, -1);
	blockDirs[9] = make_int3(0, 1, 0);
	blockDirs[10] = make_int3(0, 0, 0);
	blockDirs[11] = make_int3(0, 0, 0);
	// Dual node idx:  3
	blockDirs[12] = make_int3(0, 0, 0);
	blockDirs[13] = make_int3(0, 0, 0);
	blockDirs[14] = make_int3(0, 0, 0);
	blockDirs[15] = make_int3(0, 0, 0);
	// Dual node idx:  4
	blockDirs[16] = make_int3(0, 0, 0);
	blockDirs[17] = make_int3(1, 0, 0);
	blockDirs[18] = make_int3(0, 0, 0);
	blockDirs[19] = make_int3(0, 0, 0);
	// Dual node idx:  5
	blockDirs[20] = make_int3(1, 0, 0);
	blockDirs[21] = make_int3(0, 0, 0);
	blockDirs[22] = make_int3(0, 0, -1);
	blockDirs[23] = make_int3(0, 0, 0);
	// Dual node idx:  6
	blockDirs[24] = make_int3(0, 0, 0);
	blockDirs[25] = make_int3(0, 0, 0);
	blockDirs[26] = make_int3(0, 0, 0);
	blockDirs[27] = make_int3(0, 0, 0);
	// Dual node idx:  7
	blockDirs[28] = make_int3(0, 0, 0);
	blockDirs[29] = make_int3(0, 0, 0);
	blockDirs[30] = make_int3(0, 0, 0);
	blockDirs[31] = make_int3(0, 0, 0);
	// Dual node idx:  8
	blockDirs[32] = make_int3(0, 0, 0);
	blockDirs[33] = make_int3(0, 0, 0);
	blockDirs[34] = make_int3(0, 0, 0);
	blockDirs[35] = make_int3(0, 0, 0);
	// Dual node idx:  9
	blockDirs[36] = make_int3(0, 0, 0);
	blockDirs[37] = make_int3(0, 0, 0);
	blockDirs[38] = make_int3(0, 0, 0);
	blockDirs[39] = make_int3(0, 0, 0);
	// Dual node idx:  10
	blockDirs[40] = make_int3(0, 0, 0);
	blockDirs[41] = make_int3(0, 0, 0);
	blockDirs[42] = make_int3(0, 0, 0);
	blockDirs[43] = make_int3(0, 0, 0);
	// Dual node idx:  11
	blockDirs[44] = make_int3(0, 0, 0);
	blockDirs[45] = make_int3(0, 0, 0);
	blockDirs[46] = make_int3(0, 0, 0);
	blockDirs[47] = make_int3(0, 0, 0);
	// Dual node idx:  12
	blockDirs[48] = make_int3(0, -1, 0);
	blockDirs[49] = make_int3(0, 0, 0);
	blockDirs[50] = make_int3(0, 0, 0);
	blockDirs[51] = make_int3(-1, 0, 0);
	// Dual node idx:  13
	blockDirs[52] = make_int3(0, 0, 0);
	blockDirs[53] = make_int3(0, 0, 0);
	blockDirs[54] = make_int3(0, 0, 0);
	blockDirs[55] = make_int3(0, 0, 0);
	// Dual node idx:  14
	blockDirs[56] = make_int3(0, 0, 0);
	blockDirs[57] = make_int3(0, 0, 0);
	blockDirs[58] = make_int3(0, 0, -1);
	blockDirs[59] = make_int3(0, 0, 0);
	// Dual node idx:  15
	blockDirs[60] = make_int3(0, 0, 0);
	blockDirs[61] = make_int3(0, 0, 0);
	blockDirs[62] = make_int3(0, 0, 0);
	blockDirs[63] = make_int3(0, 0, 0);
	// Dual node idx:  16
	blockDirs[64] = make_int3(0, 0, 0);
	blockDirs[65] = make_int3(0, 0, 0);
	blockDirs[66] = make_int3(0, 0, 0);
	blockDirs[67] = make_int3(0, 0, 0);
	// Dual node idx:  17
	blockDirs[68] = make_int3(0, 0, 0);
	blockDirs[69] = make_int3(0, 0, 0);
	blockDirs[70] = make_int3(0, 0, 0);
	blockDirs[71] = make_int3(0, 0, 0);
	// Dual node idx:  18
	blockDirs[72] = make_int3(0, 0, 0);
	blockDirs[73] = make_int3(0, 0, 0);
	blockDirs[74] = make_int3(0, 0, 0);
	blockDirs[75] = make_int3(0, 0, 0);
	// Dual node idx:  19
	blockDirs[76] = make_int3(0, 0, 0);
	blockDirs[77] = make_int3(0, 0, 0);
	blockDirs[78] = make_int3(0, 0, 0);
	blockDirs[79] = make_int3(0, 0, 0);
	// Dual node idx:  20
	blockDirs[80] = make_int3(0, 0, 0);
	blockDirs[81] = make_int3(0, 0, 0);
	blockDirs[82] = make_int3(0, 0, 0);
	blockDirs[83] = make_int3(0, 0, 0);
	// Dual node idx:  21
	blockDirs[84] = make_int3(0, 0, -1);
	blockDirs[85] = make_int3(0, 0, 0);
	blockDirs[86] = make_int3(-1, 0, 0);
	blockDirs[87] = make_int3(0, 0, 0);
	// Dual node idx:  22
	blockDirs[88] = make_int3(0, 0, 0);
	blockDirs[89] = make_int3(0, 0, 0);
	blockDirs[90] = make_int3(0, 0, -1);
	blockDirs[91] = make_int3(0, 0, 0);
	// Dual node idx:  23
	blockDirs[92] = make_int3(0, 0, 0);
	blockDirs[93] = make_int3(0, 0, 0);
	blockDirs[94] = make_int3(-1, 0, 0);
	blockDirs[95] = make_int3(0, 0, 0);
	// Dual node idx:  24
	blockDirs[96] = make_int3(0, 0, 0);
	blockDirs[97] = make_int3(0, 0, 0);
	blockDirs[98] = make_int3(0, 0, 0);
	blockDirs[99] = make_int3(0, -1, 0);
	// Dual node idx:  25
	blockDirs[100] = make_int3(0, 0, 0);
	blockDirs[101] = make_int3(0, 0, 0);
	blockDirs[102] = make_int3(0, 0, 0);
	blockDirs[103] = make_int3(-1, 0, 0);
	// Dual node idx:  26
	blockDirs[104] = make_int3(0, 0, 0);
	blockDirs[105] = make_int3(0, 0, 0);
	blockDirs[106] = make_int3(0, 0, 0);
	blockDirs[107] = make_int3(0, 0, -1);
	// Dual node idx:  27
	blockDirs[108] = make_int3(0, 0, 0);
	blockDirs[109] = make_int3(0, 0, 0);
	blockDirs[110] = make_int3(0, 0, 0);
	blockDirs[111] = make_int3(0, 0, 0);
	// Dual node idx:  28
	blockDirs[112] = make_int3(0, 0, 0);
	blockDirs[113] = make_int3(0, 0, 0);
	blockDirs[114] = make_int3(0, -1, 0);
	blockDirs[115] = make_int3(0, 0, -1);
	// Dual node idx:  29
	blockDirs[116] = make_int3(0, 0, 0);
	blockDirs[117] = make_int3(0, 0, 1);
	blockDirs[118] = make_int3(0, 0, 0);
	blockDirs[119] = make_int3(0, 0, 0);
	// Dual node idx:  30
	blockDirs[120] = make_int3(0, 0, 0);
	blockDirs[121] = make_int3(0, 0, 0);
	blockDirs[122] = make_int3(0, 0, 0);
	blockDirs[123] = make_int3(0, 0, 0);
	// Dual node idx:  31
	blockDirs[124] = make_int3(0, 0, 0);
	blockDirs[125] = make_int3(0, 0, 0);
	blockDirs[126] = make_int3(0, 0, 0);
	blockDirs[127] = make_int3(-1, 0, 0);
	// Dual node idx:  32
	blockDirs[128] = make_int3(0, 0, 0);
	blockDirs[129] = make_int3(0, 0, 0);
	blockDirs[130] = make_int3(0, 0, 0);
	blockDirs[131] = make_int3(0, 0, 0);
	// Dual node idx:  33
	blockDirs[132] = make_int3(0, 0, 0);
	blockDirs[133] = make_int3(0, 0, 0);
	blockDirs[134] = make_int3(0, 0, 0);
	blockDirs[135] = make_int3(-1, 0, 0);
	// Dual node idx:  34
	blockDirs[136] = make_int3(0, 0, 0);
	blockDirs[137] = make_int3(0, 0, 0);
	blockDirs[138] = make_int3(0, 0, 0);
	blockDirs[139] = make_int3(0, 0, 0);
	// Dual node idx:  35
	blockDirs[140] = make_int3(0, 0, 0);
	blockDirs[141] = make_int3(0, 0, 0);
	blockDirs[142] = make_int3(0, 0, 0);
	blockDirs[143] = make_int3(0, 0, 0);
	// Dual node idx:  36
	blockDirs[144] = make_int3(0, 0, 0);
	blockDirs[145] = make_int3(0, 0, 0);
	blockDirs[146] = make_int3(0, 0, 0);
	blockDirs[147] = make_int3(0, 0, 0);
	// Dual node idx:  37
	blockDirs[148] = make_int3(0, 0, 0);
	blockDirs[149] = make_int3(0, 0, 0);
	blockDirs[150] = make_int3(0, 0, 0);
	blockDirs[151] = make_int3(0, -1, 0);
	// Dual node idx:  38
	blockDirs[152] = make_int3(0, 0, 0);
	blockDirs[153] = make_int3(0, 0, 0);
	blockDirs[154] = make_int3(0, 0, 0);
	blockDirs[155] = make_int3(0, -1, 0);
	// Dual node idx:  39
	blockDirs[156] = make_int3(0, 0, 0);
	blockDirs[157] = make_int3(0, 0, 0);
	blockDirs[158] = make_int3(0, 0, 0);
	blockDirs[159] = make_int3(0, -1, 0);
	// Dual node idx:  40
	blockDirs[160] = make_int3(0, 0, 0);
	blockDirs[161] = make_int3(0, 0, 0);
	blockDirs[162] = make_int3(0, 0, 0);
	blockDirs[163] = make_int3(0, 0, 0);
	// Dual node idx:  41
	blockDirs[164] = make_int3(0, 0, 0);
	blockDirs[165] = make_int3(0, 0, 0);
	blockDirs[166] = make_int3(0, 0, 0);
	blockDirs[167] = make_int3(0, 0, 0);
	// Dual node idx:  42
	blockDirs[168] = make_int3(0, 0, 0);
	blockDirs[169] = make_int3(0, 0, 0);
	blockDirs[170] = make_int3(0, 0, 0);
	blockDirs[171] = make_int3(0, 0, -1);
	// Dual node idx:  43
	blockDirs[172] = make_int3(0, 0, 0);
	blockDirs[173] = make_int3(0, 0, 0);
	blockDirs[174] = make_int3(0, 0, 0);
	blockDirs[175] = make_int3(-1, 0, 0);
	// Dual node idx:  44
	blockDirs[176] = make_int3(0, 0, 0);
	blockDirs[177] = make_int3(1, 0, 0);
	blockDirs[178] = make_int3(0, 0, 0);
	blockDirs[179] = make_int3(0, 0, 0);
	// Dual node idx:  45
	blockDirs[180] = make_int3(0, 0, 0);
	blockDirs[181] = make_int3(1, 0, 0);
	blockDirs[182] = make_int3(0, 0, 0);
	blockDirs[183] = make_int3(0, 0, 0);
	// Dual node idx:  46
	blockDirs[184] = make_int3(0, 0, 0);
	blockDirs[185] = make_int3(0, 0, 0);
	blockDirs[186] = make_int3(0, 0, 0);
	blockDirs[187] = make_int3(0, 0, 0);
	// Dual node idx:  47
	blockDirs[188] = make_int3(0, 0, 0);
	blockDirs[189] = make_int3(0, 0, 0);
	blockDirs[190] = make_int3(-1, 0, 0);
	blockDirs[191] = make_int3(0, 0, 0);
	// Dual node idx:  48
	blockDirs[192] = make_int3(0, 0, 0);
	blockDirs[193] = make_int3(0, 0, 0);
	blockDirs[194] = make_int3(0, -1, 0);
	blockDirs[195] = make_int3(0, 0, 0);
	// Dual node idx:  49
	blockDirs[196] = make_int3(0, 0, 0);
	blockDirs[197] = make_int3(0, 0, 0);
	blockDirs[198] = make_int3(0, 0, 0);
	blockDirs[199] = make_int3(0, 0, 0);
	// Dual node idx:  50
	blockDirs[200] = make_int3(0, 0, 0);
	blockDirs[201] = make_int3(0, 0, 0);
	blockDirs[202] = make_int3(0, 0, 0);
	blockDirs[203] = make_int3(0, 0, 0);
	// Dual node idx:  51
	blockDirs[204] = make_int3(0, 0, 0);
	blockDirs[205] = make_int3(0, 0, 0);
	blockDirs[206] = make_int3(0, -1, 0);
	blockDirs[207] = make_int3(0, 0, 0);
	// Dual node idx:  52
	blockDirs[208] = make_int3(0, 0, 0);
	blockDirs[209] = make_int3(0, 0, 0);
	blockDirs[210] = make_int3(0, 0, 0);
	blockDirs[211] = make_int3(0, 0, 0);
	// Dual node idx:  53
	blockDirs[212] = make_int3(0, 0, 0);
	blockDirs[213] = make_int3(0, 0, 0);
	blockDirs[214] = make_int3(0, 0, 0);
	blockDirs[215] = make_int3(0, 0, 0);
	// Dual node idx:  54
	blockDirs[216] = make_int3(0, 0, 0);
	blockDirs[217] = make_int3(0, 0, 0);
	blockDirs[218] = make_int3(0, 0, 0);
	blockDirs[219] = make_int3(0, 0, -1);
	// Dual node idx:  55
	blockDirs[220] = make_int3(0, 0, 0);
	blockDirs[221] = make_int3(0, 0, 0);
	blockDirs[222] = make_int3(0, 0, 0);
	blockDirs[223] = make_int3(0, 0, 0);
	// Dual node idx:  56
	blockDirs[224] = make_int3(0, 0, 0);
	blockDirs[225] = make_int3(0, 0, 0);
	blockDirs[226] = make_int3(0, -1, 0);
	blockDirs[227] = make_int3(0, 0, 0);
	// Dual node idx:  57
	blockDirs[228] = make_int3(0, 0, 0);
	blockDirs[229] = make_int3(0, 0, 0);
	blockDirs[230] = make_int3(0, 0, 0);
	blockDirs[231] = make_int3(0, 0, -1);
	// Dual node idx:  58
	blockDirs[232] = make_int3(0, 0, 0);
	blockDirs[233] = make_int3(0, 0, 0);
	blockDirs[234] = make_int3(0, 0, 0);
	blockDirs[235] = make_int3(0, 0, 0);
	// Dual node idx:  59
	blockDirs[236] = make_int3(0, 0, 0);
	blockDirs[237] = make_int3(0, 0, 0);
	blockDirs[238] = make_int3(0, 0, 0);
	blockDirs[239] = make_int3(0, 0, 0);
	// Dual node idx:  60
	blockDirs[240] = make_int3(0, 0, 0);
	blockDirs[241] = make_int3(0, 0, 0);
	blockDirs[242] = make_int3(-1, 0, 0);
	blockDirs[243] = make_int3(0, 0, 0);
	// Dual node idx:  61
	blockDirs[244] = make_int3(0, 0, 0);
	blockDirs[245] = make_int3(0, 0, 0);
	blockDirs[246] = make_int3(0, 0, 0);
	blockDirs[247] = make_int3(0, 0, 0);
	// Dual node idx:  62
	blockDirs[248] = make_int3(0, 0, 0);
	blockDirs[249] = make_int3(0, 0, 0);
	blockDirs[250] = make_int3(0, 0, 0);
	blockDirs[251] = make_int3(0, 0, 0);
	// Dual node idx:  63
	blockDirs[252] = make_int3(0, 0, 0);
	blockDirs[253] = make_int3(0, 0, 0);
	blockDirs[254] = make_int3(0, 0, 0);
	blockDirs[255] = make_int3(0, 0, 0);
	// Dual node idx:  64
	blockDirs[256] = make_int3(0, 0, 0);
	blockDirs[257] = make_int3(0, 0, 0);
	blockDirs[258] = make_int3(0, 0, 0);
	blockDirs[259] = make_int3(0, 0, 0);
	// Dual node idx:  65
	blockDirs[260] = make_int3(0, 0, 0);
	blockDirs[261] = make_int3(0, 0, 0);
	blockDirs[262] = make_int3(0, 0, 0);
	blockDirs[263] = make_int3(0, 0, 0);
	// Dual node idx:  66
	blockDirs[264] = make_int3(0, 0, 0);
	blockDirs[265] = make_int3(0, 0, 0);
	blockDirs[266] = make_int3(0, 0, 0);
	blockDirs[267] = make_int3(0, 0, 0);
	// Dual node idx:  67
	blockDirs[268] = make_int3(0, 0, 0);
	blockDirs[269] = make_int3(0, 0, 0);
	blockDirs[270] = make_int3(0, 0, 0);
	blockDirs[271] = make_int3(0, 0, 0);
	// Dual node idx:  68
	blockDirs[272] = make_int3(0, 1, 0);
	blockDirs[273] = make_int3(0, 0, 0);
	blockDirs[274] = make_int3(0, 0, 0);
	blockDirs[275] = make_int3(0, 0, -1);
	// Dual node idx:  69
	blockDirs[276] = make_int3(0, 1, 0);
	blockDirs[277] = make_int3(0, 0, 1);
	blockDirs[278] = make_int3(0, 0, 0);
	blockDirs[279] = make_int3(0, 0, 0);
	// Dual node idx:  70
	blockDirs[280] = make_int3(0, 0, 0);
	blockDirs[281] = make_int3(0, 0, 0);
	blockDirs[282] = make_int3(0, 0, 0);
	blockDirs[283] = make_int3(0, 0, 0);
	// Dual node idx:  71
	blockDirs[284] = make_int3(0, 0, 0);
	blockDirs[285] = make_int3(0, 0, 0);
	blockDirs[286] = make_int3(0, 0, 0);
	blockDirs[287] = make_int3(0, 0, 0);
	// Dual node idx:  72
	blockDirs[288] = make_int3(0, -1, 0);
	blockDirs[289] = make_int3(0, 0, 0);
	blockDirs[290] = make_int3(0, 0, -1);
	blockDirs[291] = make_int3(0, 0, 0);
	// Dual node idx:  73
	blockDirs[292] = make_int3(0, -1, 0);
	blockDirs[293] = make_int3(1, 0, 0);
	blockDirs[294] = make_int3(0, 0, 0);
	blockDirs[295] = make_int3(0, 0, 0);
	// Dual node idx:  74
	blockDirs[296] = make_int3(1, 0, 0);
	blockDirs[297] = make_int3(0, 0, 0);
	blockDirs[298] = make_int3(0, -1, 0);
	blockDirs[299] = make_int3(0, 0, 0);
	// Dual node idx:  75
	blockDirs[300] = make_int3(0, 0, 0);
	blockDirs[301] = make_int3(0, 0, 0);
	blockDirs[302] = make_int3(0, 0, 0);
	blockDirs[303] = make_int3(0, 0, 0);
	// Dual node idx:  76
	blockDirs[304] = make_int3(0, 0, 0);
	blockDirs[305] = make_int3(0, 0, 0);
	blockDirs[306] = make_int3(0, 0, -1);
	blockDirs[307] = make_int3(0, 0, 0);
	// Dual node idx:  77
	blockDirs[308] = make_int3(0, 0, 0);
	blockDirs[309] = make_int3(0, 1, 0);
	blockDirs[310] = make_int3(0, 0, 0);
	blockDirs[311] = make_int3(0, 0, 0);
	// Dual node idx:  78
	blockDirs[312] = make_int3(0, 0, -1);
	blockDirs[313] = make_int3(1, 0, 0);
	blockDirs[314] = make_int3(0, 0, 0);
	blockDirs[315] = make_int3(0, 0, 0);
	// Dual node idx:  79
	blockDirs[316] = make_int3(0, 0, 0);
	blockDirs[317] = make_int3(1, 0, 0);
	blockDirs[318] = make_int3(0, 0, 0);
	blockDirs[319] = make_int3(0, 0, 0);
	// Dual node idx:  80
	blockDirs[320] = make_int3(0, 0, 0);
	blockDirs[321] = make_int3(0, 0, 0);
	blockDirs[322] = make_int3(0, 0, -1);
	blockDirs[323] = make_int3(0, 0, 0);
	// Dual node idx:  81
	blockDirs[324] = make_int3(0, 0, 0);
	blockDirs[325] = make_int3(0, 0, 0);
	blockDirs[326] = make_int3(0, 0, 0);
	blockDirs[327] = make_int3(0, 0, 0);
	// Dual node idx:  82
	blockDirs[328] = make_int3(0, 0, 0);
	blockDirs[329] = make_int3(0, 0, 0);
	blockDirs[330] = make_int3(0, 0, 0);
	blockDirs[331] = make_int3(0, 0, 0);
	// Dual node idx:  83
	blockDirs[332] = make_int3(0, 0, 0);
	blockDirs[333] = make_int3(0, 0, 0);
	blockDirs[334] = make_int3(0, 0, 0);
	blockDirs[335] = make_int3(0, 0, 0);
	// Dual node idx:  84
	blockDirs[336] = make_int3(0, 0, 0);
	blockDirs[337] = make_int3(0, 0, 0);
	blockDirs[338] = make_int3(0, 0, 0);
	blockDirs[339] = make_int3(0, 0, 0);
	// Dual node idx:  85
	blockDirs[340] = make_int3(0, 0, 0);
	blockDirs[341] = make_int3(0, 0, 0);
	blockDirs[342] = make_int3(0, 0, 0);
	blockDirs[343] = make_int3(0, 0, 0);
	// Dual node idx:  86
	blockDirs[344] = make_int3(0, 0, 0);
	blockDirs[345] = make_int3(0, 0, 0);
	blockDirs[346] = make_int3(0, 0, 0);
	blockDirs[347] = make_int3(0, 0, 0);
	// Dual node idx:  87
	blockDirs[348] = make_int3(0, 0, 0);
	blockDirs[349] = make_int3(0, 0, 0);
	blockDirs[350] = make_int3(0, 0, 0);
	blockDirs[351] = make_int3(0, 0, 0);
	// Dual node idx:  88
	blockDirs[352] = make_int3(0, 0, -1);
	blockDirs[353] = make_int3(0, 0, 0);
	blockDirs[354] = make_int3(-1, 0, 0);
	blockDirs[355] = make_int3(0, 0, 0);
	// Dual node idx:  89
	blockDirs[356] = make_int3(0, 0, 0);
	blockDirs[357] = make_int3(1, 0, 0);
	blockDirs[358] = make_int3(0, 0, 0);
	blockDirs[359] = make_int3(0, 0, 0);
	// Dual node idx:  90
	blockDirs[360] = make_int3(0, 0, 0);
	blockDirs[361] = make_int3(0, 1, 0);
	blockDirs[362] = make_int3(0, 0, 0);
	blockDirs[363] = make_int3(0, 0, 0);
	// Dual node idx:  91
	blockDirs[364] = make_int3(0, 0, 0);
	blockDirs[365] = make_int3(0, 1, 0);
	blockDirs[366] = make_int3(0, 0, 0);
	blockDirs[367] = make_int3(0, 0, 0);
	// Dual node idx:  92
	blockDirs[368] = make_int3(-1, 0, 0);
	blockDirs[369] = make_int3(0, 1, 0);
	blockDirs[370] = make_int3(0, 0, 0);
	blockDirs[371] = make_int3(0, 0, 0);
	// Dual node idx:  93
	blockDirs[372] = make_int3(0, 1, 0);
	blockDirs[373] = make_int3(0, 0, 0);
	blockDirs[374] = make_int3(0, 0, 0);
	blockDirs[375] = make_int3(-1, 0, 0);
	// Dual node idx:  94
	blockDirs[376] = make_int3(0, 1, 0);
	blockDirs[377] = make_int3(0, 0, 0);
	blockDirs[378] = make_int3(0, 0, 0);
	blockDirs[379] = make_int3(0, 0, 0);
	// Dual node idx:  95
	blockDirs[380] = make_int3(0, 0, 0);
	blockDirs[381] = make_int3(0, 0, 0);
	blockDirs[382] = make_int3(0, 0, 0);
	blockDirs[383] = make_int3(0, 0, 0);
	// Dual node idx:  96
	blockDirs[384] = make_int3(0, 0, 0);
	blockDirs[385] = make_int3(0, 1, 0);
	blockDirs[386] = make_int3(0, 0, 0);
	blockDirs[387] = make_int3(0, 0, 0);
	// Dual node idx:  97
	blockDirs[388] = make_int3(0, 0, 0);
	blockDirs[389] = make_int3(0, 1, 0);
	blockDirs[390] = make_int3(0, 0, 0);
	blockDirs[391] = make_int3(0, 0, 0);
	// Dual node idx:  98
	blockDirs[392] = make_int3(0, 0, 0);
	blockDirs[393] = make_int3(0, 0, 0);
	blockDirs[394] = make_int3(0, 0, 0);
	blockDirs[395] = make_int3(0, 0, 0);
	// Dual node idx:  99
	blockDirs[396] = make_int3(0, 1, 0);
	blockDirs[397] = make_int3(0, 0, 0);
	blockDirs[398] = make_int3(1, 0, 0);
	blockDirs[399] = make_int3(0, 0, 0);
	// Dual node idx:  100
	blockDirs[400] = make_int3(0, 0, 0);
	blockDirs[401] = make_int3(0, 0, 0);
	blockDirs[402] = make_int3(0, 0, 0);
	blockDirs[403] = make_int3(0, 0, 0);
	// Dual node idx:  101
	blockDirs[404] = make_int3(1, 0, 0);
	blockDirs[405] = make_int3(0, 1, 0);
	blockDirs[406] = make_int3(0, 0, 0);
	blockDirs[407] = make_int3(0, 0, 0);
	// Dual node idx:  102
	blockDirs[408] = make_int3(0, 0, 0);
	blockDirs[409] = make_int3(0, 1, 0);
	blockDirs[410] = make_int3(0, 0, 0);
	blockDirs[411] = make_int3(0, 0, 0);
	// Dual node idx:  103
	blockDirs[412] = make_int3(0, 1, 0);
	blockDirs[413] = make_int3(0, 0, 0);
	blockDirs[414] = make_int3(0, 0, 0);
	blockDirs[415] = make_int3(0, 0, 0);
	// Dual node idx:  104
	blockDirs[416] = make_int3(1, 0, 0);
	blockDirs[417] = make_int3(0, 0, 0);
	blockDirs[418] = make_int3(0, 0, 0);
	blockDirs[419] = make_int3(0, 0, 0);
	// Dual node idx:  105
	blockDirs[420] = make_int3(0, 0, 0);
	blockDirs[421] = make_int3(0, 0, 0);
	blockDirs[422] = make_int3(0, -1, 0);
	blockDirs[423] = make_int3(0, 0, 0);
	// Dual node idx:  106
	blockDirs[424] = make_int3(0, 0, 0);
	blockDirs[425] = make_int3(0, 0, 0);
	blockDirs[426] = make_int3(0, 0, 0);
	blockDirs[427] = make_int3(0, 0, 0);
	// Dual node idx:  107
	blockDirs[428] = make_int3(0, 0, 1);
	blockDirs[429] = make_int3(0, 0, 0);
	blockDirs[430] = make_int3(-1, 0, 0);
	blockDirs[431] = make_int3(0, 0, 0);
	// Dual node idx:  108
	blockDirs[432] = make_int3(0, 0, 0);
	blockDirs[433] = make_int3(0, 0, 0);
	blockDirs[434] = make_int3(0, 0, 0);
	blockDirs[435] = make_int3(0, 0, 0);
	// Dual node idx:  109
	blockDirs[436] = make_int3(0, 0, 0);
	blockDirs[437] = make_int3(0, 0, 0);
	blockDirs[438] = make_int3(0, 0, 0);
	blockDirs[439] = make_int3(0, 0, 0);
	// Dual node idx:  110
	blockDirs[440] = make_int3(0, 0, 0);
	blockDirs[441] = make_int3(0, 0, 0);
	blockDirs[442] = make_int3(0, 0, 0);
	blockDirs[443] = make_int3(0, 0, 0);
	// Dual node idx:  111
	blockDirs[444] = make_int3(0, 0, 0);
	blockDirs[445] = make_int3(0, 0, 0);
	blockDirs[446] = make_int3(0, 0, 0);
	blockDirs[447] = make_int3(0, 0, 0);
	// Dual node idx:  112
	blockDirs[448] = make_int3(0, 0, 0);
	blockDirs[449] = make_int3(0, 0, 0);
	blockDirs[450] = make_int3(0, 0, 0);
	blockDirs[451] = make_int3(0, 0, 0);
	// Dual node idx:  113
	blockDirs[452] = make_int3(0, -1, 0);
	blockDirs[453] = make_int3(0, 0, 1);
	blockDirs[454] = make_int3(0, 0, 0);
	blockDirs[455] = make_int3(0, 0, 0);
	// Dual node idx:  114
	blockDirs[456] = make_int3(0, 0, 0);
	blockDirs[457] = make_int3(0, 0, 0);
	blockDirs[458] = make_int3(0, 0, 0);
	blockDirs[459] = make_int3(0, 0, 0);
	// Dual node idx:  115
	blockDirs[460] = make_int3(0, -1, 0);
	blockDirs[461] = make_int3(0, 0, 0);
	blockDirs[462] = make_int3(-1, 0, 0);
	blockDirs[463] = make_int3(0, 0, 0);
	// Dual node idx:  116
	blockDirs[464] = make_int3(0, 0, 0);
	blockDirs[465] = make_int3(0, 0, 0);
	blockDirs[466] = make_int3(0, 0, 0);
	blockDirs[467] = make_int3(0, 0, 0);
	// Dual node idx:  117
	blockDirs[468] = make_int3(-1, 0, 0);
	blockDirs[469] = make_int3(0, 0, 1);
	blockDirs[470] = make_int3(0, 0, 0);
	blockDirs[471] = make_int3(0, 0, 0);
	// Dual node idx:  118
	blockDirs[472] = make_int3(0, 0, 0);
	blockDirs[473] = make_int3(0, 0, 1);
	blockDirs[474] = make_int3(0, 0, 0);
	blockDirs[475] = make_int3(0, 0, 0);
	// Dual node idx:  119
	blockDirs[476] = make_int3(0, 0, 0);
	blockDirs[477] = make_int3(0, 0, 0);
	blockDirs[478] = make_int3(-1, 0, 0);
	blockDirs[479] = make_int3(0, 0, 0);
	// Dual node idx:  120
	blockDirs[480] = make_int3(0, 0, 1);
	blockDirs[481] = make_int3(0, 0, 0);
	blockDirs[482] = make_int3(0, 0, 0);
	blockDirs[483] = make_int3(0, -1, 0);
	// Dual node idx:  121
	blockDirs[484] = make_int3(0, 0, 0);
	blockDirs[485] = make_int3(0, 0, 0);
	blockDirs[486] = make_int3(0, 0, 0);
	blockDirs[487] = make_int3(0, 0, 0);
	// Dual node idx:  122
	blockDirs[488] = make_int3(0, 0, 0);
	blockDirs[489] = make_int3(0, 0, 0);
	blockDirs[490] = make_int3(0, 0, 0);
	blockDirs[491] = make_int3(0, 0, 0);
	// Dual node idx:  123
	blockDirs[492] = make_int3(0, 0, 0);
	blockDirs[493] = make_int3(0, 0, 1);
	blockDirs[494] = make_int3(0, 0, 0);
	blockDirs[495] = make_int3(0, 0, 0);
	// Dual node idx:  124
	blockDirs[496] = make_int3(0, 0, 0);
	blockDirs[497] = make_int3(0, 0, 0);
	blockDirs[498] = make_int3(0, 0, 0);
	blockDirs[499] = make_int3(0, 0, 0);
	// Dual node idx:  125
	blockDirs[500] = make_int3(0, 0, 0);
	blockDirs[501] = make_int3(0, 0, 1);
	blockDirs[502] = make_int3(0, 0, 0);
	blockDirs[503] = make_int3(0, 0, 0);
	// Dual node idx:  126
	blockDirs[504] = make_int3(1, 0, 0);
	blockDirs[505] = make_int3(0, 0, 0);
	blockDirs[506] = make_int3(0, 0, 0);
	blockDirs[507] = make_int3(0, 0, 0);
	// Dual node idx:  127
	blockDirs[508] = make_int3(0, 0, 0);
	blockDirs[509] = make_int3(0, 0, 1);
	blockDirs[510] = make_int3(0, 0, 0);
	blockDirs[511] = make_int3(0, 0, 0);
	// Dual node idx:  128
	blockDirs[512] = make_int3(1, 0, 0);
	blockDirs[513] = make_int3(0, 0, 1);
	blockDirs[514] = make_int3(0, 0, 0);
	blockDirs[515] = make_int3(0, 0, 0);
	// Dual node idx:  129
	blockDirs[516] = make_int3(0, 0, 1);
	blockDirs[517] = make_int3(0, 0, 0);
	blockDirs[518] = make_int3(1, 0, 0);
	blockDirs[519] = make_int3(0, 0, 0);
	// Dual node idx:  130
	blockDirs[520] = make_int3(0, 0, 0);
	blockDirs[521] = make_int3(1, 0, 0);
	blockDirs[522] = make_int3(0, 0, 0);
	blockDirs[523] = make_int3(0, 0, 0);
	// Dual node idx:  131
	blockDirs[524] = make_int3(0, 0, 0);
	blockDirs[525] = make_int3(0, 0, 0);
	blockDirs[526] = make_int3(0, 0, 0);
	blockDirs[527] = make_int3(0, 0, 0);
	// Dual node idx:  132
	blockDirs[528] = make_int3(0, 0, 1);
	blockDirs[529] = make_int3(0, 0, 0);
	blockDirs[530] = make_int3(0, 0, 0);
	blockDirs[531] = make_int3(0, 0, 0);
	// Dual node idx:  133
	blockDirs[532] = make_int3(0, 0, 0);
	blockDirs[533] = make_int3(0, 0, 1);
	blockDirs[534] = make_int3(0, 0, 0);
	blockDirs[535] = make_int3(0, 0, 0);
	// Dual node idx:  134
	blockDirs[536] = make_int3(0, 0, 1);
	blockDirs[537] = make_int3(0, 0, 0);
	blockDirs[538] = make_int3(0, 1, 0);
	blockDirs[539] = make_int3(0, 0, 0);
	// Dual node idx:  135
	blockDirs[540] = make_int3(0, 0, 1);
	blockDirs[541] = make_int3(0, 0, 0);
	blockDirs[542] = make_int3(0, 0, 0);
	blockDirs[543] = make_int3(0, 0, 0);

	valueInds.resize(INDICES_PER_BLOCK);
	// Dual node idx:  0
	valueInds[0] = 61;
	valueInds[1] = 83;
	valueInds[2] = 53;
	valueInds[3] = 27;
	// Dual node idx:  1
	valueInds[4] = 4;
	valueInds[5] = 44;
	valueInds[6] = 11;
	valueInds[7] = 16;
	// Dual node idx:  2
	valueInds[8] = 69;
	valueInds[9] = 24;
	valueInds[10] = 76;
	valueInds[11] = 87;
	// Dual node idx:  3
	valueInds[12] = 116;
	valueInds[13] = 36;
	valueInds[14] = 20;
	valueInds[15] = 61;
	// Dual node idx:  4
	valueInds[16] = 130;
	valueInds[17] = 31;
	valueInds[18] = 70;
	valueInds[19] = 1;
	// Dual node idx:  5
	valueInds[20] = 21;
	valueInds[21] = 79;
	valueInds[22] = 132;
	valueInds[23] = 98;
	// Dual node idx:  6
	valueInds[24] = 17;
	valueInds[25] = 59;
	valueInds[26] = 51;
	valueInds[27] = 39;
	// Dual node idx:  7
	valueInds[28] = 96;
	valueInds[29] = 90;
	valueInds[30] = 84;
	valueInds[31] = 75;
	// Dual node idx:  8
	valueInds[32] = 133;
	valueInds[33] = 69;
	valueInds[34] = 77;
	valueInds[35] = 114;
	// Dual node idx:  9
	valueInds[36] = 81;
	valueInds[37] = 75;
	valueInds[38] = 83;
	valueInds[39] = 62;
	// Dual node idx:  10
	valueInds[40] = 18;
	valueInds[41] = 58;
	valueInds[42] = 57;
	valueInds[43] = 42;
	// Dual node idx:  11
	valueInds[44] = 104;
	valueInds[45] = 1;
	valueInds[46] = 126;
	valueInds[47] = 32;
	// Dual node idx:  12
	valueInds[48] = 93;
	valueInds[49] = 30;
	valueInds[50] = 115;
	valueInds[51] = 74;
	// Dual node idx:  13
	valueInds[52] = 75;
	valueInds[53] = 32;
	valueInds[54] = 36;
	valueInds[55] = 16;
	// Dual node idx:  14
	valueInds[56] = 22;
	valueInds[57] = 40;
	valueInds[58] = 118;
	valueInds[59] = 42;
	// Dual node idx:  15
	valueInds[60] = 59;
	valueInds[61] = 58;
	valueInds[62] = 27;
	valueInds[63] = 49;
	// Dual node idx:  16
	valueInds[64] = 62;
	valueInds[65] = 1;
	valueInds[66] = 65;
	valueInds[67] = 13;
	// Dual node idx:  17
	valueInds[68] = 6;
	valueInds[69] = 49;
	valueInds[70] = 24;
	valueInds[71] = 38;
	// Dual node idx:  18
	valueInds[72] = 10;
	valueInds[73] = 49;
	valueInds[74] = 54;
	valueInds[75] = 22;
	// Dual node idx:  19
	valueInds[76] = 53;
	valueInds[77] = 82;
	valueInds[78] = 87;
	valueInds[79] = 40;
	// Dual node idx:  20
	valueInds[80] = 3;
	valueInds[81] = 52;
	valueInds[82] = 109;
	valueInds[83] = 27;
	// Dual node idx:  21
	valueInds[84] = 107;
	valueInds[85] = 40;
	valueInds[86] = 5;
	valueInds[87] = 88;
	// Dual node idx:  22
	valueInds[88] = 14;
	valueInds[89] = 47;
	valueInds[90] = 117;
	valueInds[91] = 18;
	// Dual node idx:  23
	valueInds[92] = 47;
	valueInds[93] = 40;
	valueInds[94] = 79;
	valueInds[95] = 43;
	// Dual node idx:  24
	valueInds[96] = 48;
	valueInds[97] = 54;
	valueInds[98] = 17;
	valueInds[99] = 2;
	// Dual node idx:  25
	valueInds[100] = 33;
	valueInds[101] = 30;
	valueInds[102] = 31;
	valueInds[103] = 45;
	// Dual node idx:  26
	valueInds[104] = 57;
	valueInds[105] = 55;
	valueInds[106] = 54;
	valueInds[107] = 123;
	// Dual node idx:  27
	valueInds[108] = 0;
	valueInds[109] = 15;
	valueInds[110] = 20;
	valueInds[111] = 41;
	// Dual node idx:  28
	valueInds[112] = 72;
	valueInds[113] = 55;
	valueInds[114] = 68;
	valueInds[115] = 120;
	// Dual node idx:  29
	valueInds[116] = 133;
	valueInds[117] = 42;
	valueInds[118] = 116;
	valueInds[119] = 110;
	// Dual node idx:  30
	valueInds[120] = 112;
	valueInds[121] = 37;
	valueInds[122] = 25;
	valueInds[123] = 12;
	// Dual node idx:  31
	valueInds[124] = 25;
	valueInds[125] = 109;
	valueInds[126] = 41;
	valueInds[127] = 4;
	// Dual node idx:  32
	valueInds[128] = 84;
	valueInds[129] = 13;
	valueInds[130] = 111;
	valueInds[131] = 11;
	// Dual node idx:  33
	valueInds[132] = 25;
	valueInds[133] = 38;
	valueInds[134] = 46;
	valueInds[135] = 73;
	// Dual node idx:  34
	valueInds[136] = 134;
	valueInds[137] = 114;
	valueInds[138] = 107;
	valueInds[139] = 93;
	// Dual node idx:  35
	valueInds[140] = 79;
	valueInds[141] = 44;
	valueInds[142] = 95;
	valueInds[143] = 62;
	// Dual node idx:  36
	valueInds[144] = 3;
	valueInds[145] = 110;
	valueInds[146] = 106;
	valueInds[147] = 13;
	// Dual node idx:  37
	valueInds[148] = 39;
	valueInds[149] = 30;
	valueInds[150] = 38;
	valueInds[151] = 91;
	// Dual node idx:  38
	valueInds[152] = 37;
	valueInds[153] = 33;
	valueInds[154] = 17;
	valueInds[155] = 92;
	// Dual node idx:  39
	valueInds[156] = 37;
	valueInds[157] = 108;
	valueInds[158] = 6;
	valueInds[159] = 90;
	// Dual node idx:  40
	valueInds[160] = 23;
	valueInds[161] = 14;
	valueInds[162] = 21;
	valueInds[163] = 19;
	// Dual node idx:  41
	valueInds[164] = 46;
	valueInds[165] = 27;
	valueInds[166] = 43;
	valueInds[167] = 31;
	// Dual node idx:  42
	valueInds[168] = 14;
	valueInds[169] = 82;
	valueInds[170] = 10;
	valueInds[171] = 29;
	// Dual node idx:  43
	valueInds[172] = 23;
	valueInds[173] = 53;
	valueInds[174] = 41;
	valueInds[175] = 44;
	// Dual node idx:  44
	valueInds[176] = 89;
	valueInds[177] = 43;
	valueInds[178] = 35;
	valueInds[179] = 1;
	// Dual node idx:  45
	valueInds[180] = 74;
	valueInds[181] = 25;
	valueInds[182] = 56;
	valueInds[183] = 70;
	// Dual node idx:  46
	valueInds[184] = 41;
	valueInds[185] = 49;
	valueInds[186] = 47;
	valueInds[187] = 33;
	// Dual node idx:  47
	valueInds[188] = 23;
	valueInds[189] = 22;
	valueInds[190] = 78;
	valueInds[191] = 46;
	// Dual node idx:  48
	valueInds[192] = 51;
	valueInds[193] = 55;
	valueInds[194] = 97;
	valueInds[195] = 24;
	// Dual node idx:  49
	valueInds[196] = 18;
	valueInds[197] = 15;
	valueInds[198] = 17;
	valueInds[199] = 46;
	// Dual node idx:  50
	valueInds[200] = 71;
	valueInds[201] = 67;
	valueInds[202] = 64;
	valueInds[203] = 55;
	// Dual node idx:  51
	valueInds[204] = 48;
	valueInds[205] = 67;
	valueInds[206] = 96;
	valueInds[207] = 6;
	// Dual node idx:  52
	valueInds[208] = 106;
	valueInds[209] = 20;
	valueInds[210] = 108;
	valueInds[211] = 59;
	// Dual node idx:  53
	valueInds[212] = 0;
	valueInds[213] = 19;
	valueInds[214] = 60;
	valueInds[215] = 43;
	// Dual node idx:  54
	valueInds[216] = 26;
	valueInds[217] = 24;
	valueInds[218] = 18;
	valueInds[219] = 113;
	// Dual node idx:  55
	valueInds[220] = 48;
	valueInds[221] = 50;
	valueInds[222] = 28;
	valueInds[223] = 26;
	// Dual node idx:  56
	valueInds[224] = 45;
	valueInds[225] = 73;
	valueInds[226] = 102;
	valueInds[227] = 67;
	// Dual node idx:  57
	valueInds[228] = 26;
	valueInds[229] = 64;
	valueInds[230] = 10;
	valueInds[231] = 125;
	// Dual node idx:  58
	valueInds[232] = 63;
	valueInds[233] = 15;
	valueInds[234] = 83;
	valueInds[235] = 10;
	// Dual node idx:  59
	valueInds[236] = 66;
	valueInds[237] = 15;
	valueInds[238] = 52;
	valueInds[239] = 6;
	// Dual node idx:  60
	valueInds[240] = 91;
	valueInds[241] = 92;
	valueInds[242] = 89;
	valueInds[243] = 53;
	// Dual node idx:  61
	valueInds[244] = 86;
	valueInds[245] = 75;
	valueInds[246] = 0;
	valueInds[247] = 3;
	// Dual node idx:  62
	valueInds[248] = 35;
	valueInds[249] = 16;
	valueInds[250] = 63;
	valueInds[251] = 9;
	// Dual node idx:  63
	valueInds[252] = 62;
	valueInds[253] = 66;
	valueInds[254] = 64;
	valueInds[255] = 58;
	// Dual node idx:  64
	valueInds[256] = 63;
	valueInds[257] = 50;
	valueInds[258] = 80;
	valueInds[259] = 57;
	// Dual node idx:  65
	valueInds[260] = 70;
	valueInds[261] = 16;
	valueInds[262] = 66;
	valueInds[263] = 106;
	// Dual node idx:  66
	valueInds[264] = 65;
	valueInds[265] = 63;
	valueInds[266] = 67;
	valueInds[267] = 59;
	// Dual node idx:  67
	valueInds[268] = 56;
	valueInds[269] = 50;
	valueInds[270] = 66;
	valueInds[271] = 51;
	// Dual node idx:  68
	valueInds[272] = 28;
	valueInds[273] = 97;
	valueInds[274] = 98;
	valueInds[275] = 135;
	// Dual node idx:  69
	valueInds[276] = 113;
	valueInds[277] = 2;
	valueInds[278] = 8;
	valueInds[279] = 134;
	// Dual node idx:  70
	valueInds[280] = 45;
	valueInds[281] = 4;
	valueInds[282] = 122;
	valueInds[283] = 65;
	// Dual node idx:  71
	valueInds[284] = 78;
	valueInds[285] = 73;
	valueInds[286] = 72;
	valueInds[287] = 50;
	// Dual node idx:  72
	valueInds[288] = 99;
	valueInds[289] = 71;
	valueInds[290] = 129;
	valueInds[291] = 28;
	// Dual node idx:  73
	valueInds[292] = 101;
	valueInds[293] = 33;
	valueInds[294] = 56;
	valueInds[295] = 71;
	// Dual node idx:  74
	valueInds[296] = 12;
	valueInds[297] = 45;
	valueInds[298] = 103;
	valueInds[299] = 121;
	// Dual node idx:  75
	valueInds[300] = 61;
	valueInds[301] = 7;
	valueInds[302] = 9;
	valueInds[303] = 13;
	// Dual node idx:  76
	valueInds[304] = 97;
	valueInds[305] = 2;
	valueInds[306] = 133;
	valueInds[307] = 82;
	// Dual node idx:  77
	valueInds[308] = 8;
	valueInds[309] = 105;
	valueInds[310] = 94;
	valueInds[311] = 90;
	// Dual node idx:  78
	valueInds[312] = 128;
	valueInds[313] = 47;
	valueInds[314] = 80;
	valueInds[315] = 71;
	// Dual node idx:  79
	valueInds[316] = 5;
	valueInds[317] = 23;
	valueInds[318] = 80;
	valueInds[319] = 35;
	// Dual node idx:  80
	valueInds[320] = 79;
	valueInds[321] = 78;
	valueInds[322] = 127;
	valueInds[323] = 64;
	// Dual node idx:  81
	valueInds[324] = 97;
	valueInds[325] = 96;
	valueInds[326] = 95;
	valueInds[327] = 9;
	// Dual node idx:  82
	valueInds[328] = 83;
	valueInds[329] = 19;
	valueInds[330] = 76;
	valueInds[331] = 42;
	// Dual node idx:  83
	valueInds[332] = 9;
	valueInds[333] = 0;
	valueInds[334] = 82;
	valueInds[335] = 58;
	// Dual node idx:  84
	valueInds[336] = 94;
	valueInds[337] = 7;
	valueInds[338] = 103;
	valueInds[339] = 32;
	// Dual node idx:  85
	valueInds[340] = 117;
	valueInds[341] = 113;
	valueInds[342] = 115;
	valueInds[343] = 112;
	// Dual node idx:  86
	valueInds[344] = 91;
	valueInds[345] = 90;
	valueInds[346] = 114;
	valueInds[347] = 61;
	// Dual node idx:  87
	valueInds[348] = 92;
	valueInds[349] = 2;
	valueInds[350] = 88;
	valueInds[351] = 19;
	// Dual node idx:  88
	valueInds[352] = 134;
	valueInds[353] = 87;
	valueInds[354] = 99;
	valueInds[355] = 21;
	// Dual node idx:  89
	valueInds[356] = 100;
	valueInds[357] = 60;
	valueInds[358] = 104;
	valueInds[359] = 44;
	// Dual node idx:  90
	valueInds[360] = 77;
	valueInds[361] = 39;
	valueInds[362] = 86;
	valueInds[363] = 7;
	// Dual node idx:  91
	valueInds[364] = 93;
	valueInds[365] = 37;
	valueInds[366] = 86;
	valueInds[367] = 60;
	// Dual node idx:  92
	valueInds[368] = 101;
	valueInds[369] = 38;
	valueInds[370] = 60;
	valueInds[371] = 87;
	// Dual node idx:  93
	valueInds[372] = 12;
	valueInds[373] = 91;
	valueInds[374] = 34;
	valueInds[375] = 104;
	// Dual node idx:  94
	valueInds[376] = 120;
	valueInds[377] = 77;
	valueInds[378] = 135;
	valueInds[379] = 84;
	// Dual node idx:  95
	valueInds[380] = 98;
	valueInds[381] = 100;
	valueInds[382] = 81;
	valueInds[383] = 35;
	// Dual node idx:  96
	valueInds[384] = 102;
	valueInds[385] = 51;
	valueInds[386] = 81;
	valueInds[387] = 7;
	// Dual node idx:  97
	valueInds[388] = 68;
	valueInds[389] = 48;
	valueInds[390] = 81;
	valueInds[391] = 76;
	// Dual node idx:  98
	valueInds[392] = 99;
	valueInds[393] = 95;
	valueInds[394] = 68;
	valueInds[395] = 5;
	// Dual node idx:  99
	valueInds[396] = 72;
	valueInds[397] = 101;
	valueInds[398] = 88;
	valueInds[399] = 98;
	// Dual node idx:  100
	valueInds[400] = 102;
	valueInds[401] = 101;
	valueInds[402] = 89;
	valueInds[403] = 95;
	// Dual node idx:  101
	valueInds[404] = 92;
	valueInds[405] = 73;
	valueInds[406] = 100;
	valueInds[407] = 99;
	// Dual node idx:  102
	valueInds[408] = 100;
	valueInds[409] = 56;
	valueInds[410] = 103;
	valueInds[411] = 96;
	// Dual node idx:  103
	valueInds[412] = 74;
	valueInds[413] = 102;
	valueInds[414] = 104;
	valueInds[415] = 84;
	// Dual node idx:  104
	valueInds[416] = 93;
	valueInds[417] = 89;
	valueInds[418] = 103;
	valueInds[419] = 11;
	// Dual node idx:  105
	valueInds[420] = 123;
	valueInds[421] = 113;
	valueInds[422] = 77;
	valueInds[423] = 108;
	// Dual node idx:  106
	valueInds[424] = 124;
	valueInds[425] = 36;
	valueInds[426] = 52;
	valueInds[427] = 65;
	// Dual node idx:  107
	valueInds[428] = 21;
	valueInds[429] = 118;
	valueInds[430] = 126;
	valueInds[431] = 34;
	// Dual node idx:  108
	valueInds[432] = 52;
	valueInds[433] = 112;
	valueInds[434] = 105;
	valueInds[435] = 39;
	// Dual node idx:  109
	valueInds[436] = 20;
	valueInds[437] = 112;
	valueInds[438] = 119;
	valueInds[439] = 31;
	// Dual node idx:  110
	valueInds[440] = 125;
	valueInds[441] = 29;
	valueInds[442] = 111;
	valueInds[443] = 36;
	// Dual node idx:  111
	valueInds[444] = 135;
	valueInds[445] = 110;
	valueInds[446] = 132;
	valueInds[447] = 32;
	// Dual node idx:  112
	valueInds[448] = 109;
	valueInds[449] = 108;
	valueInds[450] = 85;
	valueInds[451] = 30;
	// Dual node idx:  113
	valueInds[452] = 69;
	valueInds[453] = 54;
	valueInds[454] = 105;
	valueInds[455] = 85;
	// Dual node idx:  114
	valueInds[456] = 34;
	valueInds[457] = 8;
	valueInds[458] = 116;
	valueInds[459] = 86;
	// Dual node idx:  115
	valueInds[460] = 134;
	valueInds[461] = 85;
	valueInds[462] = 129;
	valueInds[463] = 12;
	// Dual node idx:  116
	valueInds[464] = 118;
	valueInds[465] = 29;
	valueInds[466] = 114;
	valueInds[467] = 3;
	// Dual node idx:  117
	valueInds[468] = 128;
	valueInds[469] = 22;
	valueInds[470] = 119;
	valueInds[471] = 85;
	// Dual node idx:  118
	valueInds[472] = 107;
	valueInds[473] = 14;
	valueInds[474] = 119;
	valueInds[475] = 116;
	// Dual node idx:  119
	valueInds[476] = 118;
	valueInds[477] = 117;
	valueInds[478] = 130;
	valueInds[479] = 109;
	// Dual node idx:  120
	valueInds[480] = 28;
	valueInds[481] = 123;
	valueInds[482] = 121;
	valueInds[483] = 94;
	// Dual node idx:  121
	valueInds[484] = 129;
	valueInds[485] = 122;
	valueInds[486] = 120;
	valueInds[487] = 74;
	// Dual node idx:  122
	valueInds[488] = 121;
	valueInds[489] = 131;
	valueInds[490] = 124;
	valueInds[491] = 70;
	// Dual node idx:  123
	valueInds[492] = 120;
	valueInds[493] = 26;
	valueInds[494] = 124;
	valueInds[495] = 105;
	// Dual node idx:  124
	valueInds[496] = 123;
	valueInds[497] = 125;
	valueInds[498] = 122;
	valueInds[499] = 106;
	// Dual node idx:  125
	valueInds[500] = 127;
	valueInds[501] = 57;
	valueInds[502] = 124;
	valueInds[503] = 110;
	// Dual node idx:  126
	valueInds[504] = 107;
	valueInds[505] = 130;
	valueInds[506] = 132;
	valueInds[507] = 11;
	// Dual node idx:  127
	valueInds[508] = 131;
	valueInds[509] = 80;
	valueInds[510] = 132;
	valueInds[511] = 125;
	// Dual node idx:  128
	valueInds[512] = 117;
	valueInds[513] = 78;
	valueInds[514] = 131;
	valueInds[515] = 129;
	// Dual node idx:  129
	valueInds[516] = 72;
	valueInds[517] = 128;
	valueInds[518] = 115;
	valueInds[519] = 121;
	// Dual node idx:  130
	valueInds[520] = 131;
	valueInds[521] = 119;
	valueInds[522] = 126;
	valueInds[523] = 4;
	// Dual node idx:  131
	valueInds[524] = 127;
	valueInds[525] = 128;
	valueInds[526] = 130;
	valueInds[527] = 122;
	// Dual node idx:  132
	valueInds[528] = 5;
	valueInds[529] = 127;
	valueInds[530] = 126;
	valueInds[531] = 111;
	// Dual node idx:  133
	valueInds[532] = 8;
	valueInds[533] = 76;
	valueInds[534] = 135;
	valueInds[535] = 29;
	// Dual node idx:  134
	valueInds[536] = 88;
	valueInds[537] = 69;
	valueInds[538] = 115;
	valueInds[539] = 34;
	// Dual node idx:  135
	valueInds[540] = 68;
	valueInds[541] = 133;
	valueInds[542] = 94;
	valueInds[543] = 111;

	hodges.resize(INDICES_PER_BLOCK);
	hodges[0] = 3.7925925925925927;
	hodges[1] = 2.0317460317460321;
	hodges[2] = 3.7925925925925932;
	hodges[3] = 2.4951267056530217;
	hodges[4] = 3.7925925925925927;
	hodges[5] = 3.7925925925925927;
	hodges[6] = 2.0317460317460321;
	hodges[7] = 2.4951267056530217;
	hodges[8] = 2.4951267056530213;
	hodges[9] = 3.7925925925925932;
	hodges[10] = 2.0317460317460321;
	hodges[11] = 3.7925925925925932;
	hodges[12] = 3.7925925925925927;
	hodges[13] = 2.4951267056530217;
	hodges[14] = 3.7925925925925927;
	hodges[15] = 2.0317460317460321;
	hodges[16] = 2.4951267056530217;
	hodges[17] = 3.7925925925925932;
	hodges[18] = 2.0317460317460321;
	hodges[19] = 3.7925925925925927;
	hodges[20] = 2.49512670565302;
	hodges[21] = 2.0317460317460321;
	hodges[22] = 3.7925925925925927;
	hodges[23] = 3.7925925925925927;
	hodges[24] = 2.0317460317460321;
	hodges[25] = 2.4951267056530217;
	hodges[26] = 3.7925925925925932;
	hodges[27] = 3.7925925925925927;
	hodges[28] = 3.7925925925925927;
	hodges[29] = 3.7925925925925932;
	hodges[30] = 2.0317460317460321;
	hodges[31] = 2.4951267056530217;
	hodges[32] = 2.6337448559670786;
	hodges[33] = 2.6337448559670795;
	hodges[34] = 2.6337448559670786;
	hodges[35] = 2.6337448559670777;
	hodges[36] = 3.7925925925925927;
	hodges[37] = 2.4951267056530217;
	hodges[38] = 3.7925925925925932;
	hodges[39] = 2.0317460317460321;
	hodges[40] = 2.0317460317460321;
	hodges[41] = 2.4951267056530217;
	hodges[42] = 3.7925925925925932;
	hodges[43] = 3.7925925925925927;
	hodges[44] = 3.7925925925925927;
	hodges[45] = 2.0317460317460321;
	hodges[46] = 3.7925925925925927;
	hodges[47] = 2.4951267056530217;
	hodges[48] = 2.994152046783626;
	hodges[49] = 2.1069958847736623;
	hodges[50] = 2.994152046783626;
	hodges[51] = 2.994152046783626;
	hodges[52] = 2.6337448559670777;
	hodges[53] = 2.6337448559670777;
	hodges[54] = 2.6337448559670777;
	hodges[55] = 2.6337448559670777;
	hodges[56] = 2.994152046783626;
	hodges[57] = 2.1069958847736623;
	hodges[58] = 2.994152046783626;
	hodges[59] = 2.994152046783626;
	hodges[60] = 2.6337448559670777;
	hodges[61] = 2.6337448559670777;
	hodges[62] = 2.6337448559670777;
	hodges[63] = 2.6337448559670777;
	hodges[64] = 2.994152046783626;
	hodges[65] = 2.994152046783626;
	hodges[66] = 2.994152046783626;
	hodges[67] = 2.1069958847736623;
	hodges[68] = 2.0317460317460321;
	hodges[69] = 2.4951267056530217;
	hodges[70] = 3.7925925925925927;
	hodges[71] = 3.7925925925925932;
	hodges[72] = 2.0317460317460321;
	hodges[73] = 2.4951267056530217;
	hodges[74] = 3.7925925925925927;
	hodges[75] = 3.7925925925925932;
	hodges[76] = 2.994152046783626;
	hodges[77] = 2.994152046783626;
	hodges[78] = 2.994152046783626;
	hodges[79] = 2.1069958847736623;
	hodges[80] = 3.7925925925925927;
	hodges[81] = 2.0317460317460321;
	hodges[82] = 3.7925925925925932;
	hodges[83] = 2.4951267056530217;
	hodges[84] = 2.994152046783626;
	hodges[85] = 2.1069958847736623;
	hodges[86] = 2.994152046783626;
	hodges[87] = 2.994152046783626;
	hodges[88] = 2.4951267056530217;
	hodges[89] = 2.0317460317460321;
	hodges[90] = 3.7925925925925932;
	hodges[91] = 3.7925925925925932;
	hodges[92] = 2.994152046783626;
	hodges[93] = 2.1069958847736623;
	hodges[94] = 2.994152046783626;
	hodges[95] = 2.994152046783626;
	hodges[96] = 2.4951267056530217;
	hodges[97] = 2.0317460317460321;
	hodges[98] = 3.7925925925925927;
	hodges[99] = 3.7925925925925932;
	hodges[100] = 2.994152046783626;
	hodges[101] = 2.1069958847736623;
	hodges[102] = 2.994152046783626;
	hodges[103] = 2.994152046783626;
	hodges[104] = 2.994152046783626;
	hodges[105] = 2.1069958847736623;
	hodges[106] = 2.994152046783626;
	hodges[107] = 2.994152046783626;
	hodges[108] = 2.994152046783626;
	hodges[109] = 2.1069958847736623;
	hodges[110] = 2.994152046783626;
	hodges[111] = 2.994152046783626;
	hodges[112] = 2.994152046783626;
	hodges[113] = 2.1069958847736623;
	hodges[114] = 2.994152046783626;
	hodges[115] = 2.994152046783626;
	hodges[116] = 2.4951267056530217;
	hodges[117] = 3.7925925925925927;
	hodges[118] = 2.0317460317460321;
	hodges[119] = 3.7925925925925932;
	hodges[120] = 2.6337448559670777;
	hodges[121] = 2.6337448559670777;
	hodges[122] = 2.6337448559670782;
	hodges[123] = 2.6337448559670782;
	hodges[124] = 2.4951267056530217;
	hodges[125] = 2.0317460317460321;
	hodges[126] = 3.7925925925925927;
	hodges[127] = 3.7925925925925932;
	hodges[128] = 2.994152046783626;
	hodges[129] = 2.1069958847736623;
	hodges[130] = 2.994152046783626;
	hodges[131] = 2.994152046783626;
	hodges[132] = 2.4951267056530217;
	hodges[133] = 2.0317460317460321;
	hodges[134] = 3.7925925925925927;
	hodges[135] = 3.7925925925925932;
	hodges[136] = 2.0317460317460321;
	hodges[137] = 2.4951267056530217;
	hodges[138] = 3.7925925925925927;
	hodges[139] = 3.7925925925925927;
	hodges[140] = 3.7925925925925927;
	hodges[141] = 2.0317460317460321;
	hodges[142] = 2.4951267056530217;
	hodges[143] = 3.7925925925925932;
	hodges[144] = 2.994152046783626;
	hodges[145] = 2.994152046783626;
	hodges[146] = 2.994152046783626;
	hodges[147] = 2.1069958847736623;
	hodges[148] = 2.994152046783626;
	hodges[149] = 2.1069958847736623;
	hodges[150] = 2.994152046783626;
	hodges[151] = 2.994152046783626;
	hodges[152] = 2.4951267056530217;
	hodges[153] = 2.0317460317460321;
	hodges[154] = 3.7925925925925932;
	hodges[155] = 3.7925925925925932;
	hodges[156] = 2.4951267056530217;
	hodges[157] = 2.0317460317460321;
	hodges[158] = 3.7925925925925927;
	hodges[159] = 3.7925925925925927;
	hodges[160] = 2.6337448559670777;
	hodges[161] = 2.6337448559670777;
	hodges[162] = 2.6337448559670777;
	hodges[163] = 2.6337448559670777;
	hodges[164] = 2.0317460317460321;
	hodges[165] = 2.4951267056530217;
	hodges[166] = 3.7925925925925927;
	hodges[167] = 3.7925925925925927;
	hodges[168] = 2.4951267056530217;
	hodges[169] = 2.0317460317460321;
	hodges[170] = 3.7925925925925927;
	hodges[171] = 3.7925925925925927;
	hodges[172] = 2.4951267056530217;
	hodges[173] = 2.0317460317460321;
	hodges[174] = 3.7925925925925927;
	hodges[175] = 3.7925925925925932;
	hodges[176] = 2.4951267056530217;
	hodges[177] = 3.7925925925925932;
	hodges[178] = 2.0317460317460321;
	hodges[179] = 3.7925925925925927;
	hodges[180] = 2.0317460317460321;
	hodges[181] = 2.49512670565302;
	hodges[182] = 3.7925925925925927;
	hodges[183] = 3.7925925925925927;
	hodges[184] = 2.0317460317460321;
	hodges[185] = 2.4951267056530217;
	hodges[186] = 3.7925925925925927;
	hodges[187] = 3.7925925925925927;
	hodges[188] = 2.4951267056530217;
	hodges[189] = 2.0317460317460321;
	hodges[190] = 3.7925925925925932;
	hodges[191] = 3.7925925925925927;
	hodges[192] = 2.994152046783626;
	hodges[193] = 2.1069958847736623;
	hodges[194] = 2.994152046783626;
	hodges[195] = 2.994152046783626;
	hodges[196] = 2.994152046783626;
	hodges[197] = 2.1069958847736623;
	hodges[198] = 2.994152046783626;
	hodges[199] = 2.994152046783626;
	hodges[200] = 2.994152046783626;
	hodges[201] = 2.994152046783626;
	hodges[202] = 2.994152046783626;
	hodges[203] = 2.1069958847736623;
	hodges[204] = 2.4951267056530217;
	hodges[205] = 2.0317460317460321;
	hodges[206] = 3.7925925925925927;
	hodges[207] = 3.7925925925925932;
	hodges[208] = 3.7925925925925932;
	hodges[209] = 2.0317460317460321;
	hodges[210] = 3.7925925925925927;
	hodges[211] = 2.4951267056530217;
	hodges[212] = 3.7925925925925932;
	hodges[213] = 2.4951267056530217;
	hodges[214] = 3.7925925925925927;
	hodges[215] = 2.0317460317460321;
	hodges[216] = 2.4951267056530217;
	hodges[217] = 2.0317460317460321;
	hodges[218] = 3.7925925925925927;
	hodges[219] = 3.7925925925925932;
	hodges[220] = 2.6337448559670777;
	hodges[221] = 2.6337448559670777;
	hodges[222] = 2.6337448559670777;
	hodges[223] = 2.6337448559670777;
	hodges[224] = 3.7925925925925927;
	hodges[225] = 2.0317460317460321;
	hodges[226] = 2.4951267056530217;
	hodges[227] = 3.7925925925925932;
	hodges[228] = 2.4951267056530217;
	hodges[229] = 2.0317460317460321;
	hodges[230] = 3.7925925925925932;
	hodges[231] = 3.7925925925925927;
	hodges[232] = 2.994152046783626;
	hodges[233] = 2.1069958847736623;
	hodges[234] = 2.994152046783626;
	hodges[235] = 2.994152046783626;
	hodges[236] = 2.994152046783626;
	hodges[237] = 2.1069958847736623;
	hodges[238] = 2.994152046783626;
	hodges[239] = 2.994152046783626;
	hodges[240] = 3.7925925925925927;
	hodges[241] = 2.0317460317460321;
	hodges[242] = 2.4951267056530217;
	hodges[243] = 3.7925925925925927;
	hodges[244] = 3.7925925925925927;
	hodges[245] = 2.4951267056530217;
	hodges[246] = 3.7925925925925927;
	hodges[247] = 2.0317460317460321;
	hodges[248] = 3.7925925925925932;
	hodges[249] = 2.4951267056530217;
	hodges[250] = 3.7925925925925927;
	hodges[251] = 2.0317460317460321;
	hodges[252] = 3.7925925925925927;
	hodges[253] = 2.0317460317460321;
	hodges[254] = 3.7925925925925927;
	hodges[255] = 2.4951267056530217;
	hodges[256] = 3.7925925925925927;
	hodges[257] = 2.4951267056530217;
	hodges[258] = 3.7925925925925932;
	hodges[259] = 2.0317460317460321;
	hodges[260] = 3.7925925925925932;
	hodges[261] = 2.4951267056530217;
	hodges[262] = 3.7925925925925927;
	hodges[263] = 2.0317460317460321;
	hodges[264] = 3.7925925925925927;
	hodges[265] = 2.0317460317460321;
	hodges[266] = 3.7925925925925927;
	hodges[267] = 2.4951267056530217;
	hodges[268] = 3.7925925925925932;
	hodges[269] = 2.4951267056530217;
	hodges[270] = 3.7925925925925927;
	hodges[271] = 2.0317460317460321;
	hodges[272] = 2.49512670565302;
	hodges[273] = 2.0317460317460321;
	hodges[274] = 3.7925925925925932;
	hodges[275] = 3.7925925925925927;
	hodges[276] = 2.9941520467836247;
	hodges[277] = 2.9941520467836247;
	hodges[278] = 2.1069958847736632;
	hodges[279] = 2.9941520467836265;
	hodges[280] = 3.7925925925925927;
	hodges[281] = 2.0317460317460321;
	hodges[282] = 2.4951267056530217;
	hodges[283] = 3.7925925925925932;
	hodges[284] = 3.7925925925925932;
	hodges[285] = 3.7925925925925932;
	hodges[286] = 2.0317460317460321;
	hodges[287] = 2.4951267056530217;
	hodges[288] = 3.7925925925925927;
	hodges[289] = 2.0317460317460321;
	hodges[290] = 3.7925925925925927;
	hodges[291] = 2.4951267056530217;
	hodges[292] = 2.4951267056530213;
	hodges[293] = 3.7925925925925932;
	hodges[294] = 2.0317460317460321;
	hodges[295] = 3.7925925925925932;
	hodges[296] = 2.49512670565302;
	hodges[297] = 2.0317460317460321;
	hodges[298] = 3.7925925925925927;
	hodges[299] = 3.7925925925925927;
	hodges[300] = 2.994152046783626;
	hodges[301] = 2.994152046783626;
	hodges[302] = 2.994152046783626;
	hodges[303] = 2.1069958847736623;
	hodges[304] = 3.7925925925925932;
	hodges[305] = 2.0317460317460321;
	hodges[306] = 2.4951267056530217;
	hodges[307] = 3.7925925925925927;
	hodges[308] = 2.1069958847736627;
	hodges[309] = 2.9941520467836242;
	hodges[310] = 2.994152046783626;
	hodges[311] = 2.994152046783626;
	hodges[312] = 2.4951267056530213;
	hodges[313] = 3.7925925925925932;
	hodges[314] = 2.0317460317460321;
	hodges[315] = 3.7925925925925932;
	hodges[316] = 2.0317460317460321;
	hodges[317] = 2.49512670565302;
	hodges[318] = 3.7925925925925927;
	hodges[319] = 3.7925925925925927;
	hodges[320] = 3.7925925925925927;
	hodges[321] = 2.0317460317460321;
	hodges[322] = 2.4951267056530217;
	hodges[323] = 3.7925925925925932;
	hodges[324] = 3.7925925925925927;
	hodges[325] = 2.0317460317460321;
	hodges[326] = 2.4951267056530217;
	hodges[327] = 3.7925925925925927;
	hodges[328] = 3.7925925925925927;
	hodges[329] = 2.4951267056530217;
	hodges[330] = 3.7925925925925927;
	hodges[331] = 2.0317460317460321;
	hodges[332] = 3.7925925925925932;
	hodges[333] = 2.0317460317460321;
	hodges[334] = 3.7925925925925927;
	hodges[335] = 2.4951267056530217;
	hodges[336] = 3.7925925925925927;
	hodges[337] = 2.0317460317460321;
	hodges[338] = 3.7925925925925932;
	hodges[339] = 2.4951267056530217;
	hodges[340] = 3.7925925925925932;
	hodges[341] = 3.7925925925925932;
	hodges[342] = 2.0317460317460321;
	hodges[343] = 2.4951267056530217;
	hodges[344] = 3.7925925925925932;
	hodges[345] = 2.0317460317460321;
	hodges[346] = 2.4951267056530217;
	hodges[347] = 3.7925925925925927;
	hodges[348] = 3.7925925925925932;
	hodges[349] = 3.7925925925925932;
	hodges[350] = 2.0317460317460321;
	hodges[351] = 2.4951267056530217;
	hodges[352] = 3.7925925925925927;
	hodges[353] = 2.0317460317460321;
	hodges[354] = 3.7925925925925927;
	hodges[355] = 2.4951267056530217;
	hodges[356] = 2.1069958847736627;
	hodges[357] = 2.9941520467836242;
	hodges[358] = 2.994152046783626;
	hodges[359] = 2.994152046783626;
	hodges[360] = 2.4951267056530217;
	hodges[361] = 3.7925925925925927;
	hodges[362] = 2.0317460317460321;
	hodges[363] = 3.7925925925925932;
	hodges[364] = 2.0317460317460321;
	hodges[365] = 2.49512670565302;
	hodges[366] = 3.7925925925925932;
	hodges[367] = 3.7925925925925927;
	hodges[368] = 2.4951267056530217;
	hodges[369] = 3.7925925925925932;
	hodges[370] = 2.0317460317460321;
	hodges[371] = 3.7925925925925932;
	hodges[372] = 2.49512670565302;
	hodges[373] = 2.0317460317460321;
	hodges[374] = 3.7925925925925927;
	hodges[375] = 3.7925925925925932;
	hodges[376] = 3.7925925925925927;
	hodges[377] = 2.4951267056530217;
	hodges[378] = 2.0317460317460321;
	hodges[379] = 3.7925925925925927;
	hodges[380] = 2.994152046783626;
	hodges[381] = 2.1069958847736623;
	hodges[382] = 2.994152046783626;
	hodges[383] = 2.994152046783626;
	hodges[384] = 2.4951267056530217;
	hodges[385] = 3.7925925925925927;
	hodges[386] = 2.0317460317460321;
	hodges[387] = 3.7925925925925927;
	hodges[388] = 2.0317460317460321;
	hodges[389] = 2.49512670565302;
	hodges[390] = 3.7925925925925927;
	hodges[391] = 3.7925925925925932;
	hodges[392] = 2.0317460317460321;
	hodges[393] = 2.4951267056530217;
	hodges[394] = 3.7925925925925932;
	hodges[395] = 3.7925925925925927;
	hodges[396] = 3.7925925925925927;
	hodges[397] = 2.4951267056530222;
	hodges[398] = 3.7925925925925927;
	hodges[399] = 2.0317460317460321;
	hodges[400] = 2.6337448559670786;
	hodges[401] = 2.6337448559670795;
	hodges[402] = 2.6337448559670786;
	hodges[403] = 2.6337448559670777;
	hodges[404] = 2.9941520467836247;
	hodges[405] = 2.9941520467836247;
	hodges[406] = 2.1069958847736632;
	hodges[407] = 2.9941520467836265;
	hodges[408] = 2.1069958847736627;
	hodges[409] = 2.9941520467836242;
	hodges[410] = 2.994152046783626;
	hodges[411] = 2.994152046783626;
	hodges[412] = 3.7925925925925927;
	hodges[413] = 2.4951267056530217;
	hodges[414] = 2.0317460317460321;
	hodges[415] = 3.7925925925925932;
	hodges[416] = 3.7925925925925932;
	hodges[417] = 2.4951267056530217;
	hodges[418] = 2.0317460317460321;
	hodges[419] = 3.7925925925925927;
	hodges[420] = 3.7925925925925932;
	hodges[421] = 2.0317460317460321;
	hodges[422] = 2.4951267056530217;
	hodges[423] = 3.7925925925925927;
	hodges[424] = 3.7925925925925927;
	hodges[425] = 2.4951267056530217;
	hodges[426] = 3.7925925925925932;
	hodges[427] = 2.0317460317460321;
	hodges[428] = 2.49512670565302;
	hodges[429] = 2.0317460317460321;
	hodges[430] = 3.7925925925925932;
	hodges[431] = 3.7925925925925927;
	hodges[432] = 3.7925925925925927;
	hodges[433] = 2.4951267056530217;
	hodges[434] = 3.7925925925925927;
	hodges[435] = 2.0317460317460321;
	hodges[436] = 3.7925925925925932;
	hodges[437] = 2.4951267056530217;
	hodges[438] = 3.7925925925925927;
	hodges[439] = 2.0317460317460321;
	hodges[440] = 3.7925925925925927;
	hodges[441] = 3.7925925925925932;
	hodges[442] = 2.0317460317460321;
	hodges[443] = 2.4951267056530217;
	hodges[444] = 3.7925925925925927;
	hodges[445] = 2.0317460317460321;
	hodges[446] = 3.7925925925925932;
	hodges[447] = 2.4951267056530217;
	hodges[448] = 2.994152046783626;
	hodges[449] = 2.994152046783626;
	hodges[450] = 2.994152046783626;
	hodges[451] = 2.1069958847736623;
	hodges[452] = 2.4951267056530217;
	hodges[453] = 3.7925925925925932;
	hodges[454] = 2.0317460317460321;
	hodges[455] = 3.7925925925925932;
	hodges[456] = 2.994152046783626;
	hodges[457] = 2.1069958847736623;
	hodges[458] = 2.994152046783626;
	hodges[459] = 2.994152046783626;
	hodges[460] = 3.7925925925925927;
	hodges[461] = 2.0317460317460321;
	hodges[462] = 3.7925925925925927;
	hodges[463] = 2.4951267056530217;
	hodges[464] = 3.7925925925925932;
	hodges[465] = 2.0317460317460321;
	hodges[466] = 2.4951267056530217;
	hodges[467] = 3.7925925925925927;
	hodges[468] = 2.4951267056530217;
	hodges[469] = 3.7925925925925932;
	hodges[470] = 2.0317460317460321;
	hodges[471] = 3.7925925925925932;
	hodges[472] = 2.0317460317460321;
	hodges[473] = 2.49512670565302;
	hodges[474] = 3.7925925925925927;
	hodges[475] = 3.7925925925925932;
	hodges[476] = 3.7925925925925927;
	hodges[477] = 2.0317460317460321;
	hodges[478] = 2.4951267056530217;
	hodges[479] = 3.7925925925925927;
	hodges[480] = 2.49512670565302;
	hodges[481] = 2.0317460317460321;
	hodges[482] = 3.7925925925925932;
	hodges[483] = 3.7925925925925927;
	hodges[484] = 2.0317460317460321;
	hodges[485] = 2.4951267056530217;
	hodges[486] = 3.7925925925925932;
	hodges[487] = 3.7925925925925927;
	hodges[488] = 2.994152046783626;
	hodges[489] = 2.1069958847736623;
	hodges[490] = 2.994152046783626;
	hodges[491] = 2.994152046783626;
	hodges[492] = 2.0317460317460321;
	hodges[493] = 2.49512670565302;
	hodges[494] = 3.7925925925925927;
	hodges[495] = 3.7925925925925932;
	hodges[496] = 3.7925925925925927;
	hodges[497] = 2.0317460317460321;
	hodges[498] = 2.4951267056530217;
	hodges[499] = 3.7925925925925927;
	hodges[500] = 2.4951267056530217;
	hodges[501] = 3.7925925925925927;
	hodges[502] = 2.0317460317460321;
	hodges[503] = 3.7925925925925927;
	hodges[504] = 3.7925925925925932;
	hodges[505] = 2.4951267056530217;
	hodges[506] = 2.0317460317460321;
	hodges[507] = 3.7925925925925927;
	hodges[508] = 2.1069958847736627;
	hodges[509] = 2.9941520467836242;
	hodges[510] = 2.994152046783626;
	hodges[511] = 2.994152046783626;
	hodges[512] = 2.9941520467836247;
	hodges[513] = 2.9941520467836247;
	hodges[514] = 2.1069958847736632;
	hodges[515] = 2.9941520467836265;
	hodges[516] = 3.7925925925925927;
	hodges[517] = 2.4951267056530222;
	hodges[518] = 3.7925925925925927;
	hodges[519] = 2.0317460317460321;
	hodges[520] = 2.1069958847736627;
	hodges[521] = 2.9941520467836242;
	hodges[522] = 2.994152046783626;
	hodges[523] = 2.994152046783626;
	hodges[524] = 2.6337448559670786;
	hodges[525] = 2.6337448559670795;
	hodges[526] = 2.6337448559670786;
	hodges[527] = 2.6337448559670777;
	hodges[528] = 3.7925925925925927;
	hodges[529] = 2.4951267056530217;
	hodges[530] = 2.0317460317460321;
	hodges[531] = 3.7925925925925932;
	hodges[532] = 2.1069958847736627;
	hodges[533] = 2.9941520467836242;
	hodges[534] = 2.994152046783626;
	hodges[535] = 2.994152046783626;
	hodges[536] = 3.7925925925925927;
	hodges[537] = 2.4951267056530222;
	hodges[538] = 3.7925925925925927;
	hodges[539] = 2.0317460317460321;
	hodges[540] = 3.7925925925925927;
	hodges[541] = 2.4951267056530217;
	hodges[542] = 2.0317460317460321;
	hodges[543] = 3.7925925925925927;

	return 3.7925925925925932;
}
