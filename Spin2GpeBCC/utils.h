#ifndef UTILS
#define UTILS

#include <iostream>

#include <cuda_runtime.h>
#include "helper_cuda.h"

#include "Output/Picture.hpp"
#include "Output/Text.hpp"
#include "Types/Complex.hpp"

#include "mesh.h"

// Arithmetic operators for cuda vector types
__host__ __device__ __inline__ myFloat2 operator+(myFloat2 a, myFloat2 b)
{
	return { a.x + b.x, a.y + b.y };
}
__host__ __device__ __inline__ myFloat3 operator+(myFloat3 a, myFloat3 b)
{
	return { a.x + b.x, a.y + b.y, a.z + b.z };
}
__host__ __device__ __inline__ myFloat2 operator-(myFloat2 a, myFloat2 b)
{
	return { a.x - b.x, a.y - b.y };
}
__host__ __device__ __inline__ myFloat3 operator-(myFloat3 a, myFloat3 b)
{
	return { a.x - b.x, a.y - b.y, a.z - b.z };
}
__host__ __device__ __inline__ myFloat2 operator-(myFloat2 a)
{
	return { -a.x, -a.y };
}
__host__ __device__ __inline__ void operator+=(myFloat2& a, myFloat2 b)
{
	a.x += b.x;
	a.y += b.y;
}
__host__ __device__ __inline__ void operator+=(myFloat3& a, myFloat3 b)
{
	a.x += b.x;
	a.y += b.y;
	a.z += b.z;
}
__host__ __device__ __inline__ void operator-=(myFloat2& a, myFloat2 b)
{
	a.x -= b.x;
	a.y -= b.y;
}
__host__ __device__ __inline__ myFloat2 operator*(myFloat b, myFloat2 a)
{
	return { b * a.x, b * a.y };
}
__host__ __device__ __inline__ myFloat2 operator*(myFloat2 a, myFloat b)
{
	return { b * a.x, b * a.y };
}
__host__ __device__ __inline__ myFloat3 operator*(myFloat b, myFloat3 a)
{
	return { b * a.x, b * a.y, b * a.z };
}
__host__ __device__ __inline__ myFloat3 operator*(myFloat3 a, myFloat b)
{
	return { b * a.x, b * a.y, b * a.z };
}
__host__ __device__ __inline__ myFloat3 operator/(myFloat3 a, myFloat b)
{
	return { a.x / b, a.y / b, a.z / b };
}
__host__ __device__ __inline__ myFloat2 operator/(myFloat2 a, myFloat b)
{
	return { a.x / b, a.y / b };
}
__host__ __device__ __inline__ myFloat2 conj(myFloat2 a) // Complex conjugate
{
	return { a.x, -a.y };
}
__host__ __device__ __inline__ myFloat dot(myFloat3 a, myFloat3 b) // Complex conjugate
{
	return a.x * b.x + a.y * b.y + a.z * b.z;
}
__host__ __device__ __inline__ myFloat3 cross(myFloat3 a, myFloat3 b)
{
	return {
		a.y * b.z - a.z * b.y,
		a.z * b.x - a.x * b.z,
		a.x * b.y - a.y * b.x
	};
}
__host__ __device__ __inline__ myFloat mag(myFloat3 a) // Complex conjugate
{
	return sqrt(a.x * a.x + a.y * a.y + a.z * a.z);
}
__host__ __device__ __inline__ myFloat2 operator*(myFloat2 a, myFloat2 b) // Complex number multiplication
{
	return { a.x * b.x - a.y * b.y, a.y * b.x + a.x * b.y };
}

__host__ __device__ __inline__ myFloat scalarTripleProd(myFloat3 a, myFloat3 b, myFloat3 c)
{
	return dot(a, cross(b, c));
}

__host__ __device__ __inline__  myFloat4 baryCoords(myFloat3 a, myFloat3 b, myFloat3 c, myFloat3 d, myFloat3 p)
{
	myFloat3 vap = p - a;
	myFloat3 vbp = p - b;

	myFloat3 vab = b - a;
	myFloat3 vac = c - a;
	myFloat3 vad = d - a;

	myFloat3 vbc = c - b;
	myFloat3 vbd = d - b;

	myFloat va6 = scalarTripleProd(vbp, vbd, vbc);
	myFloat vb6 = scalarTripleProd(vap, vac, vad);
	myFloat vc6 = scalarTripleProd(vap, vad, vab);
	myFloat vd6 = scalarTripleProd(vap, vab, vac);
	myFloat v6 = 1 / scalarTripleProd(vab, vac, vad);

	return { va6 * v6, vb6 * v6, vc6 * v6, vd6 * v6 };
}

__host__ __device__ __inline__ myFloat subscript(myFloat4 vec, int idx)
{
	switch (idx)
	{
	case 0:
		return vec.x;
	case 1:
		return vec.y;
	case 2:
		return vec.z;
	case 3:
		return vec.w;
	}
	return vec.x;
}

struct Complex5Vec
{
	myFloat2 s2 = {0, 0};
	myFloat2 s1 = { 0, 0 };
	myFloat2 s0 = { 0, 0 };
	myFloat2 s_1 = { 0, 0 };
	myFloat2 s_2 = { 0, 0 };
};

struct BlockPsis
{
	Complex5Vec values[VALUES_IN_BLOCK];
};

struct PitchedPtr
{
	char* __restrict__ ptr;
	size_t pitch;
	size_t slicePitch;
};

struct MagFields
{
	myFloat Bq{};
	myFloat3 Bb{};
	myFloat BqQuad{};
	myFloat3 BbQuad{};
};

std::string toString(const myFloat value, int precision = 18)
{
	std::ostringstream out;
	out.precision(precision);
	out << std::fixed << value;
	return out.str();
};

void printBasis()
{
#if BASIS == Z_QUANTIZED
	std::cout << "Using z-quantized basis!" << std::endl;
#elif BASIS == Y_QUANTIZED
	std::cout << "Using y-quantized basis!" << std::endl;
#elif BASIS == X_QUANTIZED
	std::cout << "Using x-quantized basis!" << std::endl;
#endif
}

void drawIandR(const std::string& folder, BlockPsis* h_evenPsi, size_t dxsize, size_t dysize, size_t dzsize, myFloat t, MagFields Bs, const myFloat3 p0, myFloat block_scale)
{
	const int SIZE = 2;
	const myFloat INTENSITY = 1;
	const myFloat MAG_ZERO = 0.195;
	const int width = dxsize * SIZE, height = dysize * SIZE, depth = dzsize * SIZE;
	Picture pic1(width * 5, height * 2);

	// XZ-plane
	for (uint k = 0; k < depth; ++k)
	{
		for (uint i = 0; i < width; i++)
		{
			myFloat2 norm_s2 =  {0, 0};
			myFloat2 norm_s1 =  {0, 0};
			myFloat2 norm_s0 =  {0, 0};
			myFloat2 norm_s_1 = {0, 0};
			myFloat2 norm_s_2 = {0, 0};
			myFloat minB = 99999999999999.9;
			for (uint j = 0; j < height; j++)
			{
				const uint idx = (k / SIZE) * dxsize * dysize + (j / SIZE) * dxsize + i / SIZE;
				for (uint dualNode = 0; dualNode < VALUES_IN_BLOCK; ++dualNode)
				{
					norm_s2 +=  {h_evenPsi[idx].values[dualNode].s2.x * h_evenPsi[idx].values[dualNode].s2.x  , h_evenPsi[idx].values[dualNode].s2.y * h_evenPsi[idx].values[dualNode].s2.y  };
					norm_s1 +=  {h_evenPsi[idx].values[dualNode].s1.x * h_evenPsi[idx].values[dualNode].s1.x  , h_evenPsi[idx].values[dualNode].s1.y * h_evenPsi[idx].values[dualNode].s1.y  };
					norm_s0 +=  {h_evenPsi[idx].values[dualNode].s0.x * h_evenPsi[idx].values[dualNode].s0.x  , h_evenPsi[idx].values[dualNode].s0.y * h_evenPsi[idx].values[dualNode].s0.y  };
					norm_s_1 += {h_evenPsi[idx].values[dualNode].s_1.x * h_evenPsi[idx].values[dualNode].s_1.x, h_evenPsi[idx].values[dualNode].s_1.y * h_evenPsi[idx].values[dualNode].s_1.y};
					norm_s_2 += {h_evenPsi[idx].values[dualNode].s_2.x * h_evenPsi[idx].values[dualNode].s_2.x, h_evenPsi[idx].values[dualNode].s_2.y * h_evenPsi[idx].values[dualNode].s_2.y};

					//if ((j / SIZE) == dysize / 2)
					{
						myFloat3 localPos = getLocalPos(dualNode);
						const myFloat3 globalPos = { p0.x + block_scale * (((i - 1) / SIZE) * BLOCK_WIDTH_X + localPos.x),
													p0.y + block_scale * (((j - 1) / SIZE) * BLOCK_WIDTH_Y + localPos.y),
													p0.z + block_scale * (((k - 1) / SIZE) * BLOCK_WIDTH_Z + localPos.z) };

						//myFloat3 B = magneticField(globalPos, Bs.Bq, Bs.Bz);
						//minB = min(minB, sqrt(B.x * B.x + B.y * B.y + B.z * B.z));
					}
				}
			}
			//std::cout << minB << std::endl;
			if (minB < MAG_ZERO)
			{
				pic1.setColor(i, k, Vector4(1, 0, 0, 1.0));
				pic1.setColor(width + i, k, Vector4(1, 0, 0, 1.0));
				pic1.setColor(2 * width + i, k, Vector4(1, 0, 0, 1.0));
			}
			else
			{
				const myFloat2 s2 = INTENSITY * norm_s2;
				const myFloat2 s1 = INTENSITY * norm_s1;
				const myFloat2 s0 = INTENSITY * norm_s0;
				const myFloat2 s_1 = INTENSITY * norm_s_1;
				const myFloat2 s_2 = INTENSITY * norm_s_2;
				pic1.setColor(i, k, Vector4(s2.x, s2.y, 0.0, 1.0));
				pic1.setColor(width + i, k, Vector4(s1.x, s1.y, 0.0, 1.0));
				pic1.setColor(2 * width + i, k, Vector4(s0.x, s0.y, 0.0, 1.0));
				pic1.setColor(3 * width + i, k, Vector4(s_1.x, s_1.y, 0.0, 1.0));
				pic1.setColor(4 * width + i, k, Vector4(s_2.x, s_2.y, 0.0, 1.0));
			}
		}
	}

	// XY-plane
	for (uint j = 0; j < height; j++)
	{
		for (uint i = 0; i < width; i++)
		{
			myFloat2 norm_s2 = { 0, 0 };
			myFloat2 norm_s1 = { 0, 0 };
			myFloat2 norm_s0 = { 0, 0 };
			myFloat2 norm_s_1 = { 0, 0 };
			myFloat2 norm_s_2 = { 0, 0 };
			myFloat minB = 99999999999999.9;
			for (uint k = 0; k < depth; ++k)
			{
				const uint idx = (k / SIZE) * dxsize * dysize + (j / SIZE) * dxsize + i / SIZE;
				for (uint dualNode = 0; dualNode < VALUES_IN_BLOCK; ++dualNode)
				{
					norm_s2 +=  {h_evenPsi[idx].values[dualNode].s2.x * h_evenPsi[idx].values[dualNode].s2.x  , h_evenPsi[idx].values[dualNode].s2.y * h_evenPsi[idx].values[dualNode].s2.y  };
					norm_s1 +=  {h_evenPsi[idx].values[dualNode].s1.x * h_evenPsi[idx].values[dualNode].s1.x  , h_evenPsi[idx].values[dualNode].s1.y * h_evenPsi[idx].values[dualNode].s1.y  };
					norm_s0 +=  {h_evenPsi[idx].values[dualNode].s0.x * h_evenPsi[idx].values[dualNode].s0.x  , h_evenPsi[idx].values[dualNode].s0.y * h_evenPsi[idx].values[dualNode].s0.y  };
					norm_s_1 += {h_evenPsi[idx].values[dualNode].s_1.x * h_evenPsi[idx].values[dualNode].s_1.x, h_evenPsi[idx].values[dualNode].s_1.y * h_evenPsi[idx].values[dualNode].s_1.y};
					norm_s_2 += {h_evenPsi[idx].values[dualNode].s_2.x * h_evenPsi[idx].values[dualNode].s_2.x, h_evenPsi[idx].values[dualNode].s_2.y * h_evenPsi[idx].values[dualNode].s_2.y};

					//if ((k / SIZE) == dzsize / 2)
					{
						myFloat3 localPos = getLocalPos(dualNode);
						const myFloat3 globalPos = { p0.x + block_scale * (((i - 1) / SIZE) * BLOCK_WIDTH_X + localPos.x),
													p0.y + block_scale * (((j - 1) / SIZE) * BLOCK_WIDTH_Y + localPos.y),
													p0.z + block_scale * (((k - 1) / SIZE) * BLOCK_WIDTH_Z + localPos.z) };

						//myFloat3 B = magneticField(globalPos, Bs.Bq, Bs.Bz);
						//minB = min(minB, sqrt(B.x * B.x + B.y * B.y + B.z * B.z));
					}
				}
			}
			if (minB < MAG_ZERO)
			{
				pic1.setColor(i, height + j, Vector4(1, 0, 0, 1.0));
				pic1.setColor(width + i, height + j, Vector4(1, 0, 0, 1.0));
				pic1.setColor(2 * width + i, height + j, Vector4(1, 0, 0, 1.0));
			}
			else
			{
				const myFloat2 s2 = INTENSITY * norm_s2;
				const myFloat2 s1 = INTENSITY * norm_s1;
				const myFloat2 s0 = INTENSITY * norm_s0;
				const myFloat2 s_1 = INTENSITY * norm_s_1;
				const myFloat2 s_2 = INTENSITY * norm_s_2;

				pic1.setColor(i, height + j, Vector4(s2.x, s2.y, 0.0, 1.0));
				pic1.setColor(width + i, height + j, Vector4(s1.x, s1.y, 0.0, 1.0));
				pic1.setColor(2 * width + i, height + j, Vector4(s0.x, s0.y, 0.0, 1.0));
				pic1.setColor(3 * width + i, height + j, Vector4(s_1.x, s_1.y, 0.0, 1.0));
				pic1.setColor(4 * width + i, height + j, Vector4(s_2.x, s_2.y, 0.0, 1.0));
			}
		}
	}

	for (int x = 0; x < width * 5; ++x)
	{
		pic1.setColor(x, height, Vector4(0.5, 0.5, 0.5, 1.0));
	}
	for (int y = 0; y < height * 2; ++y)
	{
		pic1.setColor(width, y, Vector4(0.5, 0.5, 0.5, 1.0));
		pic1.setColor(2 * width, y, Vector4(0.5, 0.5, 0.5, 1.0));
		pic1.setColor(3 * width, y, Vector4(0.5, 0.5, 0.5, 1.0));
		pic1.setColor(4 * width, y, Vector4(0.5, 0.5, 0.5, 1.0));
	}

	//uint axisOffsetX = 5;
	//uint axisOffsetY = 5;
	//Picture xzAxis;
	//Picture xyAxis;
	//xzAxis.load("xz_axis.bmp");
	//xyAxis.load("xy_axis.bmp");
	//for (uint x = 0; x < 60; ++x)
	//{
	//	for (uint y = 0; y < 61; ++y)
	//	{
	//		Vector4 color = xzAxis.getColor(x, y);
	//		pic1.setColor(axisOffsetX + x, axisOffsetY + y, color);
	//
	//		color = xyAxis.getColor(x, y);
	//		pic1.setColor(axisOffsetX + x, height + axisOffsetY + y, color);
	//	}
	//}

	pic1.save(folder + "/" + toString(t) + "ms.bmp", false);
	//pic1.save("mag_pos.bmp", false);
}

void drawDensity(const std::string& folder, BlockPsis* h_evenPsi, size_t dxsize, size_t dysize, size_t dzsize, myFloat t, MagFields /*Bs*/, const myFloat3 /*p0*/, myFloat /*block_scale*/)
{
	const int SIZE = 2;
	myFloat INTENSITY = 1;
	const myFloat MAG_ZERO = 0.195;
	const int width = dxsize * SIZE, height = dysize * SIZE, depth = dzsize * SIZE;
	Picture pic1(width * 5, height * 3);

	myFloat maxVal = 0;
	// XZ-plane
	for (uint k = 0; k < depth; ++k)
	{
		for (uint i = 0; i < width; i++)
		{
			myFloat norm_s2 = 0;
			myFloat norm_s1 = 0;
			myFloat norm_s0 = 0;
			myFloat norm_s_1 = 0;
			myFloat norm_s_2 = 0;
			myFloat minB = 99999999999999.9;
			for (uint j = 0; j < height; j++)
			{
				const uint idx = (k / SIZE) * dxsize * dysize + (j / SIZE) * dxsize + i / SIZE;
				for (uint dualNode = 0; dualNode < VALUES_IN_BLOCK; ++dualNode)
				{
					myFloat2 s2 = h_evenPsi[idx].values[dualNode].s2;
					myFloat2 s1 = h_evenPsi[idx].values[dualNode].s1;
					myFloat2 s0 = h_evenPsi[idx].values[dualNode].s0;
					myFloat2 s_1 = h_evenPsi[idx].values[dualNode].s_1;
					myFloat2 s_2 = h_evenPsi[idx].values[dualNode].s_2;

#if BASIS == X_QUANTIZED
					myFloat c = sqrt(6) * 0.25;
					myFloat2 x_s2 = 0.25 * s2 + 0.5 * s1 + c * s0 + 0.5 * s_1 + 0.25 * s_2;
					myFloat2 x_s1 = -0.5 * s2 - 0.5 * s1 + 0.5 * s_1 + 0.5 * s_2;
					myFloat2 x_s0 = c * s2 - 0.5 * s0 + c * s_2;
					myFloat2 x_s_1 = -0.5 * s2 + 0.5 * s1 - 0.5 * s_1 + 0.5 * s_2;
					myFloat2 x_s_2 = 0.25 * s2 - 0.5 * s1 + c * s0 - 0.5 * s_1 + 0.25 * s_2;

					s2 = x_s2;
					s1 = x_s1;
					s0 = x_s0;
					s_1 = x_s_1;
					s_2 = x_s_2;
#elif BASIS == Y_QUANTIZED
					myFloat c = sqrt(6) * 0.25;
					myFloat2 im = { 0, 1 };
					myFloat2 y_s2 = 0.25 * s2 - im * 0.5 * s1 - c * s0 + im * 0.5 * s_1 + 0.25 * s_2;
					myFloat2 y_s1 = -im * 0.5 * s2 - 0.5 * s1 - 0.5 * s_1 + im * 0.5 * s_2;
					myFloat2 y_s0 = -c * s2 - 0.5 * s0 - c * s_2;
					myFloat2 y_s_1 = im * 0.5 * s2 - 0.5 * s1 - 0.5 * s_1 - im * 0.5 * s_2;
					myFloat2 y_s_2 = 0.25 * s2 + im * 0.5 * s1 - c * s0 - im * 0.5 * s_1 + 0.25 * s_2;

					s2 = y_s2;
					s1 = y_s1;
					s0 = y_s0;
					s_1 = y_s_1;
					s_2 = y_s_2;
#endif

					norm_s2 += s2.x * s2.x + s2.y * s2.y;
					norm_s1 += s1.x * s1.x + s1.y * s1.y;
					norm_s0 += s0.x * s0.x + s0.y * s0.y;
					norm_s_1 += s_1.x * s_1.x + s_1.y * s_1.y;
					norm_s_2 += s_2.x * s_2.x + s_2.y * s_2.y;
				}
			}
			maxVal = std::max(maxVal, std::max(norm_s2, std::max(norm_s1, std::max(norm_s0, std::max(norm_s_1, norm_s_2)))));
		}
	}
	INTENSITY = 1.0 / maxVal;

	// XZ-plane
	for (uint k = 0; k < depth; ++k)
	{
		for (uint i = 0; i < width; i++)
		{
			myFloat norm_s2 = 0;
			myFloat norm_s1 = 0;
			myFloat norm_s0 = 0;
			myFloat norm_s_1 = 0;
			myFloat norm_s_2 = 0;
			myFloat minB = 99999999999999.9;
			for (uint j = 0; j < height; j++)
			{
				const uint idx = (k / SIZE) * dxsize * dysize + (j / SIZE) * dxsize + i / SIZE;
				for (uint dualNode = 0; dualNode < VALUES_IN_BLOCK; ++dualNode)
				{
					myFloat2 s2  = h_evenPsi[idx].values[dualNode].s2;
					myFloat2 s1  = h_evenPsi[idx].values[dualNode].s1;
					myFloat2 s0  = h_evenPsi[idx].values[dualNode].s0;
					myFloat2 s_1 = h_evenPsi[idx].values[dualNode].s_1;
					myFloat2 s_2 = h_evenPsi[idx].values[dualNode].s_2;

#if BASIS == X_QUANTIZED
					myFloat c = sqrt(6) * 0.25;
					myFloat2 x_s2  = 0.25 * s2 + 0.5 * s1 +   c * s0 + 0.5 * s_1 + 0.25 * s_2;
					myFloat2 x_s1  = -0.5 * s2 - 0.5 * s1            + 0.5 * s_1 +  0.5 * s_2;
					myFloat2 x_s0  =    c * s2            - 0.5 * s0             +    c * s_2;
					myFloat2 x_s_1 = -0.5 * s2 + 0.5 * s1            - 0.5 * s_1 +  0.5 * s_2;
					myFloat2 x_s_2 = 0.25 * s2 - 0.5 * s1 +   c * s0 - 0.5 * s_1 + 0.25 * s_2;

					s2 =  x_s2;
					s1 =  x_s1;
					s0 =  x_s0;
					s_1 = x_s_1;
					s_2 = x_s_2;
#elif BASIS == Y_QUANTIZED
					myFloat c = sqrt(6) * 0.25;
					myFloat2 im = { 0, 1 };
					myFloat2 y_s2  =      0.25 * s2 - im * 0.5 * s1 -   c * s0 + im * 0.5 * s_1 +     0.25 * s_2;
					myFloat2 y_s1  = -im * 0.5 * s2 -      0.5 * s1            -      0.5 * s_1 + im * 0.5 * s_2;
					myFloat2 y_s0  =        -c * s2                 - 0.5 * s0                  -        c * s_2;
					myFloat2 y_s_1 =  im * 0.5 * s2 -      0.5 * s1            -      0.5 * s_1 - im * 0.5 * s_2;
					myFloat2 y_s_2 =      0.25 * s2 + im * 0.5 * s1 -   c * s0 - im * 0.5 * s_1 +     0.25 * s_2;

					s2 =  y_s2;
					s1 =  y_s1;
					s0 =  y_s0;
					s_1 = y_s_1;
					s_2 = y_s_2;
#endif

					norm_s2 +=  s2.x *  s2.x +  s2.y *  s2.y;
					norm_s1 +=  s1.x *  s1.x +  s1.y *  s1.y;
					norm_s0 +=  s0.x *  s0.x +  s0.y *  s0.y;
					norm_s_1 += s_1.x * s_1.x + s_1.y * s_1.y;
					norm_s_2 += s_2.x * s_2.x + s_2.y * s_2.y;

					//if ((j / SIZE) == dysize / 2)
					{
						//myFloat3 localPos = getLocalPos(dualNode);
						//const myFloat3 globalPos = { p0.x + block_scale * (((i - 1) / SIZE) * BLOCK_WIDTH_X + localPos.x),
						//							p0.y + block_scale * (((j - 1) / SIZE) * BLOCK_WIDTH_Y + localPos.y),
						//							p0.z + block_scale * (((k - 1) / SIZE) * BLOCK_WIDTH_Z + localPos.z) };

						//myFloat3 B = magneticField(globalPos, Bs.Bq, Bs.Bz);
						//minB = min(minB, sqrt(B.x * B.x + B.y * B.y + B.z * B.z));
					}
				}	
			}
			//std::cout << minB << std::endl;
			if (minB < MAG_ZERO)
			{
				pic1.setColor(i, k, Vector4(1, 0, 0, 1.0));
				pic1.setColor(width + i, k, Vector4(1, 0, 0, 1.0));
				pic1.setColor(2 * width + i, k, Vector4(1, 0, 0, 1.0));
			}
			else
			{
				const myFloat s2 = INTENSITY * norm_s2;
				const myFloat s1 = INTENSITY * norm_s1;
				const myFloat s0 = INTENSITY * norm_s0;
				const myFloat s_1 = INTENSITY * norm_s_1;
				const myFloat s_2 = INTENSITY * norm_s_2;
				pic1.setColor(i,             k, Vector4(s2, s2, s2, 1.0));
				pic1.setColor(width + i,     k, Vector4(s1, s1, s1, 1.0));
				pic1.setColor(2 * width + i, k, Vector4(s0, s0, s0, 1.0));
				pic1.setColor(3 * width + i, k, Vector4(s_1, s_1, s_1, 1.0));
				pic1.setColor(4 * width + i, k, Vector4(s_2, s_2, s_2, 1.0));
			}
		}
	}

	maxVal = 0;
	// YZ-plane
	for (uint k = 0; k < depth; ++k)
	{
		for (uint j = 0; j < height; j++)
		{
			myFloat norm_s2 = 0;
			myFloat norm_s1 = 0;
			myFloat norm_s0 = 0;
			myFloat norm_s_1 = 0;
			myFloat norm_s_2 = 0;
			myFloat minB = 99999999999999.9;
			for (uint i = 0; i < width; i++)
			{
				const uint idx = (k / SIZE) * dxsize * dysize + (j / SIZE) * dxsize + i / SIZE;
				for (uint dualNode = 0; dualNode < VALUES_IN_BLOCK; ++dualNode)
				{
					myFloat2 s2 = h_evenPsi[idx].values[dualNode].s2;
					myFloat2 s1 = h_evenPsi[idx].values[dualNode].s1;
					myFloat2 s0 = h_evenPsi[idx].values[dualNode].s0;
					myFloat2 s_1 = h_evenPsi[idx].values[dualNode].s_1;
					myFloat2 s_2 = h_evenPsi[idx].values[dualNode].s_2;

#if BASIS == X_QUANTIZED
					myFloat c = sqrt(6) * 0.25;
					myFloat2 x_s2 = 0.25 * s2 + 0.5 * s1 + c * s0 + 0.5 * s_1 + 0.25 * s_2;
					myFloat2 x_s1 = -0.5 * s2 - 0.5 * s1 + 0.5 * s_1 + 0.5 * s_2;
					myFloat2 x_s0 = c * s2 - 0.5 * s0 + c * s_2;
					myFloat2 x_s_1 = -0.5 * s2 + 0.5 * s1 - 0.5 * s_1 + 0.5 * s_2;
					myFloat2 x_s_2 = 0.25 * s2 - 0.5 * s1 + c * s0 - 0.5 * s_1 + 0.25 * s_2;

					s2 = x_s2;
					s1 = x_s1;
					s0 = x_s0;
					s_1 = x_s_1;
					s_2 = x_s_2;
#elif BASIS == Y_QUANTIZED
					myFloat c = sqrt(6) * 0.25;
					myFloat2 im = { 0, 1 };
					myFloat2 y_s2 = 0.25 * s2 - im * 0.5 * s1 - c * s0 + im * 0.5 * s_1 + 0.25 * s_2;
					myFloat2 y_s1 = -im * 0.5 * s2 - 0.5 * s1 - 0.5 * s_1 + im * 0.5 * s_2;
					myFloat2 y_s0 = -c * s2 - 0.5 * s0 - c * s_2;
					myFloat2 y_s_1 = im * 0.5 * s2 - 0.5 * s1 - 0.5 * s_1 - im * 0.5 * s_2;
					myFloat2 y_s_2 = 0.25 * s2 + im * 0.5 * s1 - c * s0 - im * 0.5 * s_1 + 0.25 * s_2;

					s2 = y_s2;
					s1 = y_s1;
					s0 = y_s0;
					s_1 = y_s_1;
					s_2 = y_s_2;
#endif

					norm_s2 += s2.x * s2.x + s2.y * s2.y;
					norm_s1 += s1.x * s1.x + s1.y * s1.y;
					norm_s0 += s0.x * s0.x + s0.y * s0.y;
					norm_s_1 += s_1.x * s_1.x + s_1.y * s_1.y;
					norm_s_2 += s_2.x * s_2.x + s_2.y * s_2.y;
				}
			}
			maxVal = std::max(maxVal, std::max(norm_s2, std::max(norm_s1, std::max(norm_s0, std::max(norm_s_1, norm_s_2)))));
		}
	}
	INTENSITY = 1.0 / maxVal;

	// YZ-plane
	for (uint k = 0; k < depth; ++k)
	{
		for (uint j = 0; j < height; j++)
		{
			myFloat norm_s2 = 0;
			myFloat norm_s1 = 0;
			myFloat norm_s0 = 0;
			myFloat norm_s_1 = 0;
			myFloat norm_s_2 = 0;
			myFloat minB = 99999999999999.9;
			for (uint i = 0; i < width; i++)
			{
				const uint idx = (k / SIZE) * dxsize * dysize + (j / SIZE) * dxsize + i / SIZE;
				for (uint dualNode = 0; dualNode < VALUES_IN_BLOCK; ++dualNode)
				{
					myFloat2 s2 = h_evenPsi[idx].values[dualNode].s2;
					myFloat2 s1 = h_evenPsi[idx].values[dualNode].s1;
					myFloat2 s0 = h_evenPsi[idx].values[dualNode].s0;
					myFloat2 s_1 = h_evenPsi[idx].values[dualNode].s_1;
					myFloat2 s_2 = h_evenPsi[idx].values[dualNode].s_2;

#if BASIS == X_QUANTIZED
					myFloat c = sqrt(6) * 0.25;
					myFloat2 x_s2  = 0.25 * s2 + 0.5 * s1 +   c * s0 + 0.5 * s_1 + 0.25 * s_2;
					myFloat2 x_s1  = -0.5 * s2 - 0.5 * s1            + 0.5 * s_1 +  0.5 * s_2;
					myFloat2 x_s0  =    c * s2            - 0.5 * s0             +    c * s_2;
					myFloat2 x_s_1 = -0.5 * s2 + 0.5 * s1            - 0.5 * s_1 +  0.5 * s_2;
					myFloat2 x_s_2 = 0.25 * s2 - 0.5 * s1 +   c * s0 - 0.5 * s_1 + 0.25 * s_2;

					s2 = x_s2;
					s1 = x_s1;
					s0 = x_s0;
					s_1 = x_s_1;
					s_2 = x_s_2;
#elif BASIS == Y_QUANTIZED
					myFloat c = sqrt(6) * 0.25;
					myFloat2 im = { 0, 1 };
					myFloat2 y_s2  =      0.25 * s2 - im * 0.5 * s1 -   c * s0 + im * 0.5 * s_1 +     0.25 * s_2;
					myFloat2 y_s1  = -im * 0.5 * s2 -      0.5 * s1            -      0.5 * s_1 + im * 0.5 * s_2;
					myFloat2 y_s0  =        -c * s2                 - 0.5 * s0                  -        c * s_2;
					myFloat2 y_s_1 =  im * 0.5 * s2 -      0.5 * s1            -      0.5 * s_1 - im * 0.5 * s_2;
					myFloat2 y_s_2 =      0.25 * s2 + im * 0.5 * s1 -   c * s0 - im * 0.5 * s_1 +     0.25 * s_2;

					s2 = y_s2;
					s1 = y_s1;
					s0 = y_s0;
					s_1 = y_s_1;
					s_2 = y_s_2;
#endif

					norm_s2 += s2.x * s2.x + s2.y * s2.y;
					norm_s1 += s1.x * s1.x + s1.y * s1.y;
					norm_s0 += s0.x * s0.x + s0.y * s0.y;
					norm_s_1 += s_1.x * s_1.x + s_1.y * s_1.y;
					norm_s_2 += s_2.x * s_2.x + s_2.y * s_2.y;

					//if ((j / SIZE) == dysize / 2)
					{
						//myFloat3 localPos = getLocalPos(dualNode);
						//const myFloat3 globalPos = { p0.x + block_scale * (((i - 1) / SIZE) * BLOCK_WIDTH_X + localPos.x),
						//							p0.y + block_scale * (((j - 1) / SIZE) * BLOCK_WIDTH_Y + localPos.y),
						//							p0.z + block_scale * (((k - 1) / SIZE) * BLOCK_WIDTH_Z + localPos.z) };

						//myFloat3 B = magneticField(globalPos, Bs.Bq, Bs.Bz);
						//minB = min(minB, sqrt(B.x * B.x + B.y * B.y + B.z * B.z));
					}
				}
			}
			//std::cout << minB << std::endl;
			if (minB < MAG_ZERO)
			{
				pic1.setColor(j, k, Vector4(1, 0, 0, 1.0));
				pic1.setColor(width + j, k, Vector4(1, 0, 0, 1.0));
				pic1.setColor(2 * width + j, k, Vector4(1, 0, 0, 1.0));
			}
			else
			{
				const myFloat s2 = INTENSITY * norm_s2;
				const myFloat s1 = INTENSITY * norm_s1;
				const myFloat s0 = INTENSITY * norm_s0;
				const myFloat s_1 = INTENSITY * norm_s_1;
				const myFloat s_2 = INTENSITY * norm_s_2;
				pic1.setColor(j,             height + k, Vector4(s2, s2, s2, 1.0));
				pic1.setColor(width + j,     height + k, Vector4(s1, s1, s1, 1.0));
				pic1.setColor(2 * width + j, height + k, Vector4(s0, s0, s0, 1.0));
				pic1.setColor(3 * width + j, height + k, Vector4(s_1, s_1, s_1, 1.0));
				pic1.setColor(4 * width + j, height + k, Vector4(s_2, s_2, s_2, 1.0));
			}
		}
	}

	// XY-plane
	maxVal = 0;
	for (uint j = 0; j < height; j++)
	{
		for (uint i = 0; i < width; i++)
		{
			myFloat norm_s2 = 0;
			myFloat norm_s1 = 0;
			myFloat norm_s0 = 0;
			myFloat norm_s_1 = 0;
			myFloat norm_s_2 = 0;
			myFloat minB = 99999999999999.9;
			for (uint k = 0; k < depth; ++k)
			{
				const uint idx = (k / SIZE) * dxsize * dysize + (j / SIZE) * dxsize + i / SIZE;
				for (uint dualNode = 0; dualNode < VALUES_IN_BLOCK; ++dualNode)
				{
					myFloat2 s2 = h_evenPsi[idx].values[dualNode].s2;
					myFloat2 s1 = h_evenPsi[idx].values[dualNode].s1;
					myFloat2 s0 = h_evenPsi[idx].values[dualNode].s0;
					myFloat2 s_1 = h_evenPsi[idx].values[dualNode].s_1;
					myFloat2 s_2 = h_evenPsi[idx].values[dualNode].s_2;

#if BASIS == X_QUANTIZED
					myFloat c = sqrt(6) * 0.25;
					myFloat2 x_s2  = 0.25 * s2 + 0.5 * s1 +   c * s0 + 0.5 * s_1 + 0.25 * s_2;
					myFloat2 x_s1  = -0.5 * s2 - 0.5 * s1            + 0.5 * s_1 +  0.5 * s_2;
					myFloat2 x_s0  =    c * s2            - 0.5 * s0             +    c * s_2;
					myFloat2 x_s_1 = -0.5 * s2 + 0.5 * s1            - 0.5 * s_1 +  0.5 * s_2;
					myFloat2 x_s_2 = 0.25 * s2 - 0.5 * s1 +   c * s0 - 0.5 * s_1 + 0.25 * s_2;

					s2 = x_s2;
					s1 = x_s1;
					s0 = x_s0;
					s_1 = x_s_1;
					s_2 = x_s_2;
#elif BASIS == Y_QUANTIZED
					myFloat c = sqrt(6) * 0.25;
					myFloat2 im = { 0, 1 };
					myFloat2 y_s2  =      0.25 * s2 - im * 0.5 * s1 -   c * s0 + im * 0.5 * s_1 +     0.25 * s_2;
					myFloat2 y_s1  = -im * 0.5 * s2 -      0.5 * s1            -      0.5 * s_1 + im * 0.5 * s_2;
					myFloat2 y_s0  =        -c * s2                 - 0.5 * s0                  -        c * s_2;
					myFloat2 y_s_1 =  im * 0.5 * s2 -      0.5 * s1            -      0.5 * s_1 - im * 0.5 * s_2;
					myFloat2 y_s_2 =      0.25 * s2 + im * 0.5 * s1 -   c * s0 - im * 0.5 * s_1 +     0.25 * s_2;

					s2 = y_s2;
					s1 = y_s1;
					s0 = y_s0;
					s_1 = y_s_1;
					s_2 = y_s_2;
#endif

					norm_s2 += s2.x * s2.x + s2.y * s2.y;
					norm_s1 += s1.x * s1.x + s1.y * s1.y;
					norm_s0 += s0.x * s0.x + s0.y * s0.y;
					norm_s_1 += s_1.x * s_1.x + s_1.y * s_1.y;
					norm_s_2 += s_2.x * s_2.x + s_2.y * s_2.y;
				}
			}
			maxVal = std::max(maxVal, std::max(norm_s2, std::max(norm_s1, std::max(norm_s0, std::max(norm_s_1, norm_s_2)))));
		}
	}
	INTENSITY = 1.0 / maxVal;
	
	// XY-plane
	for (uint j = 0; j < height; j++)
	{
		for (uint i = 0; i < width; i++)
		{
			myFloat norm_s2 = 0;
			myFloat norm_s1 = 0;
			myFloat norm_s0 = 0;
			myFloat norm_s_1 = 0;
			myFloat norm_s_2 = 0;
			myFloat minB = 99999999999999.9;
			for (uint k = 0; k < depth; ++k)
			{
				const uint idx = (k / SIZE) * dxsize * dysize + (j / SIZE) * dxsize + i / SIZE;
				for (uint dualNode = 0; dualNode < VALUES_IN_BLOCK; ++dualNode)
				{
					myFloat2 s2 = h_evenPsi[idx].values[dualNode].s2;
					myFloat2 s1 = h_evenPsi[idx].values[dualNode].s1;
					myFloat2 s0 = h_evenPsi[idx].values[dualNode].s0;
					myFloat2 s_1 = h_evenPsi[idx].values[dualNode].s_1;
					myFloat2 s_2 = h_evenPsi[idx].values[dualNode].s_2;

#if BASIS == X_QUANTIZED
					myFloat c = sqrt(6) * 0.25;
					myFloat2 x_s2  = 0.25 * s2 + 0.5 * s1 +   c * s0 + 0.5 * s_1 + 0.25 * s_2;
					myFloat2 x_s1  = -0.5 * s2 - 0.5 * s1            + 0.5 * s_1 +  0.5 * s_2;
					myFloat2 x_s0  =    c * s2            - 0.5 * s0             +    c * s_2;
					myFloat2 x_s_1 = -0.5 * s2 + 0.5 * s1            - 0.5 * s_1 +  0.5 * s_2;
					myFloat2 x_s_2 = 0.25 * s2 - 0.5 * s1 +   c * s0 - 0.5 * s_1 + 0.25 * s_2;

					s2 = x_s2;
					s1 = x_s1;
					s0 = x_s0;
					s_1 = x_s_1;
					s_2 = x_s_2;
#elif BASIS == Y_QUANTIZED
					myFloat c = sqrt(6) * 0.25;
					myFloat2 im = { 0, 1 };
					myFloat2 y_s2  =      0.25 * s2 - im * 0.5 * s1 -   c * s0 + im * 0.5 * s_1 +     0.25 * s_2;
					myFloat2 y_s1  = -im * 0.5 * s2 -      0.5 * s1            -      0.5 * s_1 + im * 0.5 * s_2;
					myFloat2 y_s0  =        -c * s2                 - 0.5 * s0                  -        c * s_2;
					myFloat2 y_s_1 =  im * 0.5 * s2 -      0.5 * s1            -      0.5 * s_1 - im * 0.5 * s_2;
					myFloat2 y_s_2 =      0.25 * s2 + im * 0.5 * s1 -   c * s0 - im * 0.5 * s_1 +     0.25 * s_2;

					s2 = y_s2;
					s1 = y_s1;
					s0 = y_s0;
					s_1 = y_s_1;
					s_2 = y_s_2;
#endif

					norm_s2 += s2.x * s2.x + s2.y * s2.y;
					norm_s1 += s1.x * s1.x + s1.y * s1.y;
					norm_s0 += s0.x * s0.x + s0.y * s0.y;
					norm_s_1 += s_1.x * s_1.x + s_1.y * s_1.y;
					norm_s_2 += s_2.x * s_2.x + s_2.y * s_2.y;

					//if ((k / SIZE) == dzsize / 2)
					{
						//myFloat3 localPos = getLocalPos(dualNode);
						//const myFloat3 globalPos = { p0.x + block_scale * (((i - 1) / SIZE) * BLOCK_WIDTH_X + localPos.x),
						//							p0.y + block_scale * (((j - 1) / SIZE) * BLOCK_WIDTH_Y + localPos.y),
						//							p0.z + block_scale * (((k - 1) / SIZE) * BLOCK_WIDTH_Z + localPos.z) };

						//myFloat3 B = magneticField(globalPos, Bs.Bq, Bs.Bz);
						//minB = min(minB, sqrt(B.x * B.x + B.y * B.y + B.z * B.z));
					}
				}
			}
			if (minB < MAG_ZERO)
			{
				pic1.setColor(i, height + j, Vector4(1, 0, 0, 1.0));
				pic1.setColor(width + i, height + j, Vector4(1, 0, 0, 1.0));
				pic1.setColor(2 * width + i, height + j, Vector4(1, 0, 0, 1.0));
			}
			else
			{
				const myFloat s2 = INTENSITY * norm_s2;
				const myFloat s1 = INTENSITY * norm_s1;
				const myFloat s0 = INTENSITY * norm_s0;
				const myFloat s_1 = INTENSITY * norm_s_1;
				const myFloat s_2 = INTENSITY * norm_s_2;

				pic1.setColor(i,             2 * height + j, Vector4(s2, s2, s2, 1.0));
				pic1.setColor(width + i,     2 * height + j, Vector4(s1, s1, s1, 1.0));
				pic1.setColor(2 * width + i, 2 * height + j, Vector4(s0, s0, s0, 1.0));
				pic1.setColor(3 * width + i, 2 * height + j, Vector4(s_1, s_1, s_1, 1.0));
				pic1.setColor(4 * width + i, 2 * height + j, Vector4(s_2, s_2, s_2, 1.0));
			}
		}
	}

	for (int x = 0; x < width * 5; ++x)
	{
		pic1.setColor(x, height, Vector4(0.5, 0.5, 0.5, 1.0));
		pic1.setColor(x, 2 * height, Vector4(0.5, 0.5, 0.5, 1.0));
	}
	for (int y = 0; y < height * 3; ++y)
	{
		pic1.setColor(width, y, Vector4(0.5, 0.5, 0.5, 1.0));
		pic1.setColor(2 * width, y, Vector4(0.5, 0.5, 0.5, 1.0));
		pic1.setColor(3 * width, y, Vector4(0.5, 0.5, 0.5, 1.0));
		pic1.setColor(4 * width, y, Vector4(0.5, 0.5, 0.5, 1.0));
	}

	//uint axisOffsetX = 5;
	//uint axisOffsetY = 5;
	//Picture xzAxis;
	//Picture xyAxis;
	//xzAxis.load("xz_axis.bmp");
	//xyAxis.load("xy_axis.bmp");
	//for (uint x = 0; x < 60; ++x)
	//{
	//	for (uint y = 0; y < 61; ++y)
	//	{
	//		Vector4 color = xzAxis.getColor(x, y);
	//		pic1.setColor(axisOffsetX + x, axisOffsetY + y, color);
	//
	//		color = xyAxis.getColor(x, y);
	//		pic1.setColor(axisOffsetX + x, height + axisOffsetY + y, color);
	//	}
	//}

	pic1.save(folder + "/" + toString(t) + "ms.bmp", false);
	//pic1.save("mag_pos.bmp", false);
}

template <typename T>
void swapEnd(T& var)
{
	char* varArray = reinterpret_cast<char*>(&var);
	for (long i = 0; i < static_cast<long>(sizeof(var) / 2); i++)
		std::swap(varArray[sizeof(var) - 1 - i], varArray[i]);
}

constexpr myFloat DENSITY_THRESHOLD = 0.0001;
constexpr myFloat DISTANCE_THRESHOLD = 4;

void saveVolume(const std::string& folder, BlockPsis* pPsi, size_t bsize, size_t dxsize, size_t dysize, size_t dzsize, myFloat block_scale, myFloat3 p0, myFloat t)
{
	std::ofstream file;
	file.open(folder + "/" + std::to_string(t) + ".vtk", std::ios::out | std::ios::binary);

	file << "# vtk DataFile Version 3.0" << std::endl
	<< "Comment if needed" << std::endl;

	file << "BINARY" << std::endl;

	uint64_t pointCount = dxsize * dysize * dzsize * bsize;

	file << "DATASET POLYDATA" << std::endl << "POINTS " << pointCount << " myFloat" << std::endl;

	for (uint z = 0; z < dzsize; ++z)
	{
		for (uint x = 0; x < dxsize; ++x)
		{
			for (uint y = 0; y < dysize; ++y)
			{
				for (uint dualNode = 0; dualNode < VALUES_IN_BLOCK; ++dualNode)
				{
					myFloat3 localPos = getLocalPos(dualNode);
					myFloat3 myFloatGlobalPos = { p0.x + block_scale * (x * BLOCK_WIDTH_X + localPos.x),
						p0.y + block_scale * (y * BLOCK_WIDTH_Y + localPos.y),
						p0.z + block_scale * (z * BLOCK_WIDTH_Z + localPos.z) };
					myFloat3 globalPos = myFloat3{ (myFloat)myFloatGlobalPos.x, (myFloat)myFloatGlobalPos.y, (myFloat)myFloatGlobalPos.z };

					swapEnd(globalPos.x);
					swapEnd(globalPos.y);
					swapEnd(globalPos.z);
					
					file.write((char*)&globalPos.x, sizeof(myFloat));
					file.write((char*)&globalPos.y, sizeof(myFloat));
					file.write((char*)&globalPos.z, sizeof(myFloat));
				}
			}
		}
	}

	file << std::endl << "POINT_DATA " << pointCount << std::endl;
	file << "SCALARS m=2 myFloat 1" << std::endl;
	file << "LOOKUP_TABLE default" << std::endl;

	for (uint z = 0; z < dzsize; ++z)
	{
		for (uint x = 0; x < dxsize; ++x)
		{
			for (uint y = 0; y < dysize; ++y)
			{
				const uint idx = z * dxsize * dysize + y * dxsize + x;
				for (uint dualNode = 0; dualNode < VALUES_IN_BLOCK; ++dualNode)
				{
					myFloat2 s2 =  pPsi[idx].values[dualNode].s2;
					myFloat2 s1 =  pPsi[idx].values[dualNode].s1;
					myFloat2 s0 =  pPsi[idx].values[dualNode].s0;
					myFloat2 s_1 = pPsi[idx].values[dualNode].s_1;
					myFloat2 s_2 = pPsi[idx].values[dualNode].s_2;
#if BASIS == X_QUANTIZED
					myFloat c = sqrt(6) * 0.25;
					myFloat2 x_s2  = 0.25 * s2 + 0.5 * s1 +   c * s0 + 0.5 * s_1 + 0.25 * s_2;

					s2 = x_s2;
#elif BASIS == Y_QUANTIZED
					myFloat c = sqrt(6) * 0.25;
					myFloat2 im = { 0, 1 };
					myFloat2 y_s2  =      0.25 * s2 - im * 0.5 * s1 -   c * s0 + im * 0.5 * s_1 +     0.25 * s_2;

					s2 = y_s2;
#endif

					myFloat dens_m2 = s2.x * s2.x + s2.y * s2.y;
	
					myFloat density = (myFloat)(dens_m2);
					swapEnd(density);
					file.write((char*)&density, sizeof(myFloat));
				}
			}
		}
	}
	
	file << std::endl << "SCALARS m=1 myFloat 1" << std::endl;
	file << "LOOKUP_TABLE default" << std::endl;
	
	for (uint z = 0; z < dzsize; ++z)
	{
		for (uint x = 0; x < dxsize; ++x)
		{
			for (uint y = 0; y < dysize; ++y)
			{
				const uint idx = z * dxsize * dysize + y * dxsize + x;
				for (uint dualNode = 0; dualNode < VALUES_IN_BLOCK; ++dualNode)
				{
					myFloat2 s2 = pPsi[idx].values[dualNode].s2;
					myFloat2 s1 = pPsi[idx].values[dualNode].s1;
					myFloat2 s0 = pPsi[idx].values[dualNode].s0;
					myFloat2 s_1 = pPsi[idx].values[dualNode].s_1;
					myFloat2 s_2 = pPsi[idx].values[dualNode].s_2;
#if BASIS == X_QUANTIZED
					myFloat c = sqrt(6) * 0.25;

					myFloat2 x_s1 = -0.5 * s2 - 0.5 * s1 + 0.5 * s_1 + 0.5 * s_2;


					s1 = x_s1;
#elif BASIS == Y_QUANTIZED
					myFloat c = sqrt(6) * 0.25;
					myFloat2 im = { 0, 1 };

					myFloat2 y_s1 = -im * 0.5 * s2 - 0.5 * s1 - 0.5 * s_1 + im * 0.5 * s_2;

					s1 = y_s1;
#endif

					myFloat dens_m1 = s1.x * s1.x + s1.y * s1.y;
	
					myFloat density = (myFloat)(dens_m1);
					swapEnd(density);
					file.write((char*)&density, sizeof(myFloat));
				}
			}
		}
	}

	file << std::endl << "SCALARS m=0 myFloat 1" << std::endl;
	file << "LOOKUP_TABLE default" << std::endl;

	for (uint z = 0; z < dzsize; ++z)
	{
		for (uint x = 0; x < dxsize; ++x)
		{
			for (uint y = 0; y < dysize; ++y)
			{
				const uint idx = z * dxsize * dysize + y * dxsize + x;
				for (uint dualNode = 0; dualNode < VALUES_IN_BLOCK; ++dualNode)
				{
					myFloat2 s2 = pPsi[idx].values[dualNode].s2;
					myFloat2 s1 = pPsi[idx].values[dualNode].s1;
					myFloat2 s0 = pPsi[idx].values[dualNode].s0;
					myFloat2 s_1 = pPsi[idx].values[dualNode].s_1;
					myFloat2 s_2 = pPsi[idx].values[dualNode].s_2;
#if BASIS == X_QUANTIZED
					myFloat c = sqrt(6) * 0.25;

					myFloat2 x_s0 = c * s2 - 0.5 * s0 + c * s_2;

					s0 = x_s0;
#elif BASIS == Y_QUANTIZED
					myFloat c = sqrt(6) * 0.25;
					myFloat2 im = { 0, 1 };

					myFloat2 y_s0 = -c * s2 - 0.5 * s0 - c * s_2;

					s0 = y_s0;
#endif

					myFloat dens_m0 = s0.x * s0.x + s0.y * s0.y;

					myFloat density = (myFloat)(dens_m0);
					swapEnd(density);
					file.write((char*)&density, sizeof(myFloat));
				}
			}
		}
	}

	file << std::endl << "SCALARS m=-1 myFloat 1" << std::endl;
	file << "LOOKUP_TABLE default" << std::endl;

	for (uint z = 0; z < dzsize; ++z)
	{
		for (uint x = 0; x < dxsize; ++x)
		{
			for (uint y = 0; y < dysize; ++y)
			{
				const uint idx = z * dxsize * dysize + y * dxsize + x;
				for (uint dualNode = 0; dualNode < VALUES_IN_BLOCK; ++dualNode)
				{
					myFloat2 s2 = pPsi[idx].values[dualNode].s2;
					myFloat2 s1 = pPsi[idx].values[dualNode].s1;
					myFloat2 s0 = pPsi[idx].values[dualNode].s0;
					myFloat2 s_1 = pPsi[idx].values[dualNode].s_1;
					myFloat2 s_2 = pPsi[idx].values[dualNode].s_2;
#if BASIS == X_QUANTIZED
					myFloat c = sqrt(6) * 0.25;

					myFloat2 x_s_1 = -0.5 * s2 + 0.5 * s1 - 0.5 * s_1 + 0.5 * s_2;

					s_1 = x_s_1;
#elif BASIS == Y_QUANTIZED
					myFloat c = sqrt(6) * 0.25;
					myFloat2 im = { 0, 1 };

					myFloat2 y_s_1 = im * 0.5 * s2 - 0.5 * s1 - 0.5 * s_1 - im * 0.5 * s_2;

					s_1 = y_s_1;
#endif

					myFloat dens_m_1 = s_1.x * s_1.x + s_1.y * s_1.y;

					myFloat density = (myFloat)(dens_m_1);
					swapEnd(density);
					file.write((char*)&density, sizeof(myFloat));
				}
			}
		}
	}

	file << std::endl << "SCALARS m=-2 myFloat 1" << std::endl;
	file << "LOOKUP_TABLE default" << std::endl;

	for (uint z = 0; z < dzsize; ++z)
	{
		for (uint x = 0; x < dxsize; ++x)
		{
			for (uint y = 0; y < dysize; ++y)
			{
				const uint idx = z * dxsize * dysize + y * dxsize + x;
				for (uint dualNode = 0; dualNode < VALUES_IN_BLOCK; ++dualNode)
				{
					myFloat2 s2 = pPsi[idx].values[dualNode].s2;
					myFloat2 s1 = pPsi[idx].values[dualNode].s1;
					myFloat2 s0 = pPsi[idx].values[dualNode].s0;
					myFloat2 s_1 = pPsi[idx].values[dualNode].s_1;
					myFloat2 s_2 = pPsi[idx].values[dualNode].s_2;
#if BASIS == X_QUANTIZED
					myFloat c = sqrt(6) * 0.25;

					myFloat2 x_s_2 = 0.25 * s2 - 0.5 * s1 + c * s0 - 0.5 * s_1 + 0.25 * s_2;

					s_2 = x_s_2;
#elif BASIS == Y_QUANTIZED
					myFloat c = sqrt(6) * 0.25;
					myFloat2 im = { 0, 1 };

					myFloat2 y_s_2 = 0.25 * s2 + im * 0.5 * s1 - c * s0 - im * 0.5 * s_1 + 0.25 * s_2;

					s_2 = y_s_2;
#endif

					myFloat dens_m_2 = s_2.x * s_2.x + s_2.y * s_2.y;

					myFloat density = (myFloat)(dens_m_2);
					swapEnd(density);
					file.write((char*)&density, sizeof(myFloat));
				}
			}
		}
	}

	//file << "VERTICES " << pointCount << " " << pointCount << std::endl;
	//for (int i = 0; i < pointCount; ++i)
	//{
	//	int swapped = i;
	//	swapEnd(swapped);
	//	file.write((char*)&swapped, sizeof(int));
	//}

	file << std::endl;
	file.close();
}

void saveSpinor(const std::string& folder, BlockPsis* pPsi, size_t bsize, size_t dxsize, size_t dysize, size_t dzsize, myFloat block_scale, myFloat3 p0, myFloat t)
{
	std::ofstream file;
	file.open(folder + "/" + std::to_string(t) + ".vtk", std::ios::out | std::ios::binary);

	file << "# vtk DataFile Version 3.0" << std::endl
		<< "Comment if needed" << std::endl;

	file << "BINARY" << std::endl;

	uint64_t pointCount = dxsize * dysize * dzsize * bsize;

	file << "DATASET POLYDATA" << std::endl << "POINTS " << pointCount << " myFloat" << std::endl;

	for (uint z = 0; z < dzsize; ++z)
	{
		for (uint x = 0; x < dxsize; ++x)
		{
			for (uint y = 0; y < dysize; ++y)
			{
				for (uint dualNode = 0; dualNode < VALUES_IN_BLOCK; ++dualNode)
				{
					myFloat3 localPos = getLocalPos(dualNode);
					myFloat3 myFloatGlobalPos = { p0.x + block_scale * (x * BLOCK_WIDTH_X + localPos.x),
						p0.y + block_scale * (y * BLOCK_WIDTH_Y + localPos.y),
						p0.z + block_scale * (z * BLOCK_WIDTH_Z + localPos.z) };
					myFloat3 globalPos = myFloat3{ (myFloat)myFloatGlobalPos.x, (myFloat)myFloatGlobalPos.y, (myFloat)myFloatGlobalPos.z };

					swapEnd(globalPos.x);
					swapEnd(globalPos.y);
					swapEnd(globalPos.z);

					file.write((char*)&globalPos.x, sizeof(myFloat));
					file.write((char*)&globalPos.y, sizeof(myFloat));
					file.write((char*)&globalPos.z, sizeof(myFloat));
				}
			}
		}
	}

	file << std::endl << "POINT_DATA " << pointCount << std::endl;
	file << "SCALARS r_m=2 myFloat 1" << std::endl;
	file << "LOOKUP_TABLE default" << std::endl;

	for (uint z = 0; z < dzsize; ++z)
	{
		for (uint x = 0; x < dxsize; ++x)
		{
			for (uint y = 0; y < dysize; ++y)
			{
				const uint idx = z * dxsize * dysize + y * dxsize + x;
				for (uint dualNode = 0; dualNode < VALUES_IN_BLOCK; ++dualNode)
				{
					myFloat2 s2 = pPsi[idx].values[dualNode].s2;
					myFloat2 s1 = pPsi[idx].values[dualNode].s1;
					myFloat2 s0 = pPsi[idx].values[dualNode].s0;
					myFloat2 s_1 = pPsi[idx].values[dualNode].s_1;
					myFloat2 s_2 = pPsi[idx].values[dualNode].s_2;

#if BASIS == X_QUANTIZED
					myFloat c = sqrt(6) * 0.25;
					myFloat2 x_s2  = 0.25 * s2 + 0.5 * s1 +   c * s0 + 0.5 * s_1 + 0.25 * s_2;
					myFloat2 x_s1  = -0.5 * s2 - 0.5 * s1            + 0.5 * s_1 +  0.5 * s_2;
					myFloat2 x_s0  =    c * s2            - 0.5 * s0             +    c * s_2;
					myFloat2 x_s_1 = -0.5 * s2 + 0.5 * s1            - 0.5 * s_1 +  0.5 * s_2;
					myFloat2 x_s_2 = 0.25 * s2 - 0.5 * s1 +   c * s0 - 0.5 * s_1 + 0.25 * s_2;

					s2 = x_s2;
					s1 = x_s1;
					s0 = x_s0;
					s_1 = x_s_1;
					s_2 = x_s_2;
#elif BASIS == Y_QUANTIZED
					myFloat c = sqrt(6) * 0.25;
					myFloat2 im = { 0, 1 };
					myFloat2 y_s2  =      0.25 * s2 - im * 0.5 * s1 -   c * s0 + im * 0.5 * s_1 +     0.25 * s_2;
					myFloat2 y_s1  = -im * 0.5 * s2 -      0.5 * s1            -      0.5 * s_1 + im * 0.5 * s_2;
					myFloat2 y_s0  =        -c * s2                 - 0.5 * s0                  -        c * s_2;
					myFloat2 y_s_1 =  im * 0.5 * s2 -      0.5 * s1            -      0.5 * s_1 - im * 0.5 * s_2;
					myFloat2 y_s_2 =      0.25 * s2 + im * 0.5 * s1 -   c * s0 - im * 0.5 * s_1 +     0.25 * s_2;

					s2 = y_s2;
					s1 = y_s1;
					s0 = y_s0;
					s_1 = y_s_1;
					s_2 = y_s_2;
#endif

					myFloat dens_s2 = s2.x * s2.x + s2.y * s2.y;
					myFloat dens_s1 = s1.x * s1.x + s1.y * s1.y;
					myFloat dens_s0 = s0.x * s0.x + s0.y * s0.y;
					myFloat dens_s_1 = s_1.x * s_1.x + s_1.y * s_1.y;
					myFloat dens_s_2 = s_2.x * s_2.x + s_2.y * s_2.y;
					myFloat dens = dens_s2 + dens_s1 + dens_s0 + dens_s_1 + dens_s_2;
					
					myFloat s2_r = 0;
					if (DENSITY_THRESHOLD < dens)
						s2_r = (myFloat)(s2.x / sqrt(dens));
					swapEnd(s2_r);
					file.write((char*)&s2_r, sizeof(myFloat));
				}
			}
		}
	}
	file << std::endl << "SCALARS i_m=2 myFloat 1" << std::endl;
	file << "LOOKUP_TABLE default" << std::endl;

	for (uint z = 0; z < dzsize; ++z)
	{
		for (uint x = 0; x < dxsize; ++x)
		{
			for (uint y = 0; y < dysize; ++y)
			{
				const uint idx = z * dxsize * dysize + y * dxsize + x;
				for (uint dualNode = 0; dualNode < VALUES_IN_BLOCK; ++dualNode)
				{
					myFloat2 s2 = pPsi[idx].values[dualNode].s2;
					myFloat2 s1 = pPsi[idx].values[dualNode].s1;
					myFloat2 s0 = pPsi[idx].values[dualNode].s0;
					myFloat2 s_1 = pPsi[idx].values[dualNode].s_1;
					myFloat2 s_2 = pPsi[idx].values[dualNode].s_2;
#if BASIS == X_QUANTIZED
					myFloat c = sqrt(6) * 0.25;
					myFloat2 x_s2  = 0.25 * s2 + 0.5 * s1 +   c * s0 + 0.5 * s_1 + 0.25 * s_2;
					myFloat2 x_s1  = -0.5 * s2 - 0.5 * s1            + 0.5 * s_1 +  0.5 * s_2;
					myFloat2 x_s0  =    c * s2            - 0.5 * s0             +    c * s_2;
					myFloat2 x_s_1 = -0.5 * s2 + 0.5 * s1            - 0.5 * s_1 +  0.5 * s_2;
					myFloat2 x_s_2 = 0.25 * s2 - 0.5 * s1 +   c * s0 - 0.5 * s_1 + 0.25 * s_2;

					s2 = x_s2;
					s1 = x_s1;
					s0 = x_s0;
					s_1 = x_s_1;
					s_2 = x_s_2;
#elif BASIS == Y_QUANTIZED
					myFloat c = sqrt(6) * 0.25;
					myFloat2 im = { 0, 1 };
					myFloat2 y_s2  =      0.25 * s2 - im * 0.5 * s1 -   c * s0 + im * 0.5 * s_1 +     0.25 * s_2;
					myFloat2 y_s1  = -im * 0.5 * s2 -      0.5 * s1            -      0.5 * s_1 + im * 0.5 * s_2;
					myFloat2 y_s0  =        -c * s2                 - 0.5 * s0                  -        c * s_2;
					myFloat2 y_s_1 =  im * 0.5 * s2 -      0.5 * s1            -      0.5 * s_1 - im * 0.5 * s_2;
					myFloat2 y_s_2 =      0.25 * s2 + im * 0.5 * s1 -   c * s0 - im * 0.5 * s_1 +     0.25 * s_2;

					s2 = y_s2;
					s1 = y_s1;
					s0 = y_s0;
					s_1 = y_s_1;
					s_2 = y_s_2;
#endif

					myFloat dens_s2 = s2.x * s2.x + s2.y * s2.y;
					myFloat dens_s1 = s1.x * s1.x + s1.y * s1.y;
					myFloat dens_s0 = s0.x * s0.x + s0.y * s0.y;
					myFloat dens_s_1 = s_1.x * s_1.x + s_1.y * s_1.y;
					myFloat dens_s_2 = s_2.x * s_2.x + s_2.y * s_2.y;
					myFloat dens = dens_s2 + dens_s1 + dens_s0 + dens_s_1 + dens_s_2;

					myFloat s2_i = 0;
					if (DENSITY_THRESHOLD < dens)
						s2_i = (myFloat)(s2.y / sqrt(dens));
					swapEnd(s2_i);
					file.write((char*)&s2_i, sizeof(myFloat));
				}
			}
		}
	}

	file << std::endl << "SCALARS r_m=1 myFloat 1" << std::endl;
	file << "LOOKUP_TABLE default" << std::endl;

	for (uint z = 0; z < dzsize; ++z)
	{
		for (uint x = 0; x < dxsize; ++x)
		{
			for (uint y = 0; y < dysize; ++y)
			{
				const uint idx = z * dxsize * dysize + y * dxsize + x;
				for (uint dualNode = 0; dualNode < VALUES_IN_BLOCK; ++dualNode)
				{
					myFloat2 s2 = pPsi[idx].values[dualNode].s2;
					myFloat2 s1 = pPsi[idx].values[dualNode].s1;
					myFloat2 s0 = pPsi[idx].values[dualNode].s0;
					myFloat2 s_1 = pPsi[idx].values[dualNode].s_1;
					myFloat2 s_2 = pPsi[idx].values[dualNode].s_2;
#if BASIS == X_QUANTIZED
					myFloat c = sqrt(6) * 0.25;
					myFloat2 x_s2  = 0.25 * s2 + 0.5 * s1 +   c * s0 + 0.5 * s_1 + 0.25 * s_2;
					myFloat2 x_s1  = -0.5 * s2 - 0.5 * s1            + 0.5 * s_1 +  0.5 * s_2;
					myFloat2 x_s0  =    c * s2            - 0.5 * s0             +    c * s_2;
					myFloat2 x_s_1 = -0.5 * s2 + 0.5 * s1            - 0.5 * s_1 +  0.5 * s_2;
					myFloat2 x_s_2 = 0.25 * s2 - 0.5 * s1 +   c * s0 - 0.5 * s_1 + 0.25 * s_2;

					s2 = x_s2;
					s1 = x_s1;
					s0 = x_s0;
					s_1 = x_s_1;
					s_2 = x_s_2;
#elif BASIS == Y_QUANTIZED
					myFloat c = sqrt(6) * 0.25;
					myFloat2 im = { 0, 1 };
					myFloat2 y_s2  =      0.25 * s2 - im * 0.5 * s1 -   c * s0 + im * 0.5 * s_1 +     0.25 * s_2;
					myFloat2 y_s1  = -im * 0.5 * s2 -      0.5 * s1            -      0.5 * s_1 + im * 0.5 * s_2;
					myFloat2 y_s0  =        -c * s2                 - 0.5 * s0                  -        c * s_2;
					myFloat2 y_s_1 =  im * 0.5 * s2 -      0.5 * s1            -      0.5 * s_1 - im * 0.5 * s_2;
					myFloat2 y_s_2 =      0.25 * s2 + im * 0.5 * s1 -   c * s0 - im * 0.5 * s_1 +     0.25 * s_2;

					s2 = y_s2;
					s1 = y_s1;
					s0 = y_s0;
					s_1 = y_s_1;
					s_2 = y_s_2;
#endif

					myFloat dens_s2 = s2.x * s2.x + s2.y * s2.y;
					myFloat dens_s1 = s1.x * s1.x + s1.y * s1.y;
					myFloat dens_s0 = s0.x * s0.x + s0.y * s0.y;
					myFloat dens_s_1 = s_1.x * s_1.x + s_1.y * s_1.y;
					myFloat dens_s_2 = s_2.x * s_2.x + s_2.y * s_2.y;
					myFloat dens = dens_s2 + dens_s1 + dens_s0 + dens_s_1 + dens_s_2;

					myFloat s1_r = 0;
					if (DENSITY_THRESHOLD < dens)
						s1_r = (myFloat)(s1.x / sqrt(dens));
					swapEnd(s1_r);
					file.write((char*)&s1_r, sizeof(myFloat));
				}
			}
		}
	}
	file << std::endl << "SCALARS i_m=1 myFloat 1" << std::endl;
	file << "LOOKUP_TABLE default" << std::endl;

	for (uint z = 0; z < dzsize; ++z)
	{
		for (uint x = 0; x < dxsize; ++x)
		{
			for (uint y = 0; y < dysize; ++y)
			{
				const uint idx = z * dxsize * dysize + y * dxsize + x;
				for (uint dualNode = 0; dualNode < VALUES_IN_BLOCK; ++dualNode)
				{
					myFloat2 s2 = pPsi[idx].values[dualNode].s2;
					myFloat2 s1 = pPsi[idx].values[dualNode].s1;
					myFloat2 s0 = pPsi[idx].values[dualNode].s0;
					myFloat2 s_1 = pPsi[idx].values[dualNode].s_1;
					myFloat2 s_2 = pPsi[idx].values[dualNode].s_2;
#if BASIS == X_QUANTIZED
					myFloat c = sqrt(6) * 0.25;
					myFloat2 x_s2  = 0.25 * s2 + 0.5 * s1 +   c * s0 + 0.5 * s_1 + 0.25 * s_2;
					myFloat2 x_s1  = -0.5 * s2 - 0.5 * s1            + 0.5 * s_1 +  0.5 * s_2;
					myFloat2 x_s0  =    c * s2            - 0.5 * s0             +    c * s_2;
					myFloat2 x_s_1 = -0.5 * s2 + 0.5 * s1            - 0.5 * s_1 +  0.5 * s_2;
					myFloat2 x_s_2 = 0.25 * s2 - 0.5 * s1 +   c * s0 - 0.5 * s_1 + 0.25 * s_2;

					s2 = x_s2;
					s1 = x_s1;
					s0 = x_s0;
					s_1 = x_s_1;
					s_2 = x_s_2;
#elif BASIS == Y_QUANTIZED
					myFloat c = sqrt(6) * 0.25;
					myFloat2 im = { 0, 1 };
					myFloat2 y_s2  =      0.25 * s2 - im * 0.5 * s1 -   c * s0 + im * 0.5 * s_1 +     0.25 * s_2;
					myFloat2 y_s1  = -im * 0.5 * s2 -      0.5 * s1            -      0.5 * s_1 + im * 0.5 * s_2;
					myFloat2 y_s0  =        -c * s2                 - 0.5 * s0                  -        c * s_2;
					myFloat2 y_s_1 =  im * 0.5 * s2 -      0.5 * s1            -      0.5 * s_1 - im * 0.5 * s_2;
					myFloat2 y_s_2 =      0.25 * s2 + im * 0.5 * s1 -   c * s0 - im * 0.5 * s_1 +     0.25 * s_2;

					s2 = y_s2;
					s1 = y_s1;
					s0 = y_s0;
					s_1 = y_s_1;
					s_2 = y_s_2;
#endif

					myFloat dens_s2 = s2.x * s2.x + s2.y * s2.y;
					myFloat dens_s1 = s1.x * s1.x + s1.y * s1.y;
					myFloat dens_s0 = s0.x * s0.x + s0.y * s0.y;
					myFloat dens_s_1 = s_1.x * s_1.x + s_1.y * s_1.y;
					myFloat dens_s_2 = s_2.x * s_2.x + s_2.y * s_2.y;
					myFloat dens = dens_s2 + dens_s1 + dens_s0 + dens_s_1 + dens_s_2;

					myFloat s1_i = 0;
					if (DENSITY_THRESHOLD < dens)
						s1_i = (myFloat)(s1.y / sqrt(dens));
					swapEnd(s1_i);
					file.write((char*)&s1_i, sizeof(myFloat));
				}
			}
		}
	}

	file << std::endl << "SCALARS r_m=0 myFloat 1" << std::endl;
	file << "LOOKUP_TABLE default" << std::endl;

	for (uint z = 0; z < dzsize; ++z)
	{
		for (uint x = 0; x < dxsize; ++x)
		{
			for (uint y = 0; y < dysize; ++y)
			{
				const uint idx = z * dxsize * dysize + y * dxsize + x;
				for (uint dualNode = 0; dualNode < VALUES_IN_BLOCK; ++dualNode)
				{
					myFloat2 s2 = pPsi[idx].values[dualNode].s2;
					myFloat2 s1 = pPsi[idx].values[dualNode].s1;
					myFloat2 s0 = pPsi[idx].values[dualNode].s0;
					myFloat2 s_1 = pPsi[idx].values[dualNode].s_1;
					myFloat2 s_2 = pPsi[idx].values[dualNode].s_2;
#if BASIS == X_QUANTIZED
					myFloat c = sqrt(6) * 0.25;
					myFloat2 x_s2  = 0.25 * s2 + 0.5 * s1 +   c * s0 + 0.5 * s_1 + 0.25 * s_2;
					myFloat2 x_s1  = -0.5 * s2 - 0.5 * s1            + 0.5 * s_1 +  0.5 * s_2;
					myFloat2 x_s0  =    c * s2            - 0.5 * s0             +    c * s_2;
					myFloat2 x_s_1 = -0.5 * s2 + 0.5 * s1            - 0.5 * s_1 +  0.5 * s_2;
					myFloat2 x_s_2 = 0.25 * s2 - 0.5 * s1 +   c * s0 - 0.5 * s_1 + 0.25 * s_2;

					s2 = x_s2;
					s1 = x_s1;
					s0 = x_s0;
					s_1 = x_s_1;
					s_2 = x_s_2;
#elif BASIS == Y_QUANTIZED
					myFloat c = sqrt(6) * 0.25;
					myFloat2 im = { 0, 1 };
					myFloat2 y_s2  =      0.25 * s2 - im * 0.5 * s1 -   c * s0 + im * 0.5 * s_1 +     0.25 * s_2;
					myFloat2 y_s1  = -im * 0.5 * s2 -      0.5 * s1            -      0.5 * s_1 + im * 0.5 * s_2;
					myFloat2 y_s0  =        -c * s2                 - 0.5 * s0                  -        c * s_2;
					myFloat2 y_s_1 =  im * 0.5 * s2 -      0.5 * s1            -      0.5 * s_1 - im * 0.5 * s_2;
					myFloat2 y_s_2 =      0.25 * s2 + im * 0.5 * s1 -   c * s0 - im * 0.5 * s_1 +     0.25 * s_2;

					s2 = y_s2;
					s1 = y_s1;
					s0 = y_s0;
					s_1 = y_s_1;
					s_2 = y_s_2;
#endif

					myFloat dens_s2 = s2.x * s2.x + s2.y * s2.y;
					myFloat dens_s1 = s1.x * s1.x + s1.y * s1.y;
					myFloat dens_s0 = s0.x * s0.x + s0.y * s0.y;
					myFloat dens_s_1 = s_1.x * s_1.x + s_1.y * s_1.y;
					myFloat dens_s_2 = s_2.x * s_2.x + s_2.y * s_2.y;
					myFloat dens = dens_s2 + dens_s1 + dens_s0 + dens_s_1 + dens_s_2;

					myFloat s0_r = 0;
					if (DENSITY_THRESHOLD < dens)
						s0_r = (myFloat)(s0.x / sqrt(dens));
					swapEnd(s0_r);
					file.write((char*)&s0_r, sizeof(myFloat));
				}
			}
		}
	}
	file << std::endl << "SCALARS i_m=0 myFloat 1" << std::endl;
	file << "LOOKUP_TABLE default" << std::endl;

	for (uint z = 0; z < dzsize; ++z)
	{
		for (uint x = 0; x < dxsize; ++x)
		{
			for (uint y = 0; y < dysize; ++y)
			{
				const uint idx = z * dxsize * dysize + y * dxsize + x;
				for (uint dualNode = 0; dualNode < VALUES_IN_BLOCK; ++dualNode)
				{
					myFloat2 s2 = pPsi[idx].values[dualNode].s2;
					myFloat2 s1 = pPsi[idx].values[dualNode].s1;
					myFloat2 s0 = pPsi[idx].values[dualNode].s0;
					myFloat2 s_1 = pPsi[idx].values[dualNode].s_1;
					myFloat2 s_2 = pPsi[idx].values[dualNode].s_2;
#if BASIS == X_QUANTIZED
					myFloat c = sqrt(6) * 0.25;
					myFloat2 x_s2  = 0.25 * s2 + 0.5 * s1 +   c * s0 + 0.5 * s_1 + 0.25 * s_2;
					myFloat2 x_s1  = -0.5 * s2 - 0.5 * s1            + 0.5 * s_1 +  0.5 * s_2;
					myFloat2 x_s0  =    c * s2            - 0.5 * s0             +    c * s_2;
					myFloat2 x_s_1 = -0.5 * s2 + 0.5 * s1            - 0.5 * s_1 +  0.5 * s_2;
					myFloat2 x_s_2 = 0.25 * s2 - 0.5 * s1 +   c * s0 - 0.5 * s_1 + 0.25 * s_2;

					s2 = x_s2;
					s1 = x_s1;
					s0 = x_s0;
					s_1 = x_s_1;
					s_2 = x_s_2;
#elif BASIS == Y_QUANTIZED
					myFloat c = sqrt(6) * 0.25;
					myFloat2 im = { 0, 1 };
					myFloat2 y_s2  =      0.25 * s2 - im * 0.5 * s1 -   c * s0 + im * 0.5 * s_1 +     0.25 * s_2;
					myFloat2 y_s1  = -im * 0.5 * s2 -      0.5 * s1            -      0.5 * s_1 + im * 0.5 * s_2;
					myFloat2 y_s0  =        -c * s2                 - 0.5 * s0                  -        c * s_2;
					myFloat2 y_s_1 =  im * 0.5 * s2 -      0.5 * s1            -      0.5 * s_1 - im * 0.5 * s_2;
					myFloat2 y_s_2 =      0.25 * s2 + im * 0.5 * s1 -   c * s0 - im * 0.5 * s_1 +     0.25 * s_2;

					s2 = y_s2;
					s1 = y_s1;
					s0 = y_s0;
					s_1 = y_s_1;
					s_2 = y_s_2;
#endif

					myFloat dens_s2 = s2.x * s2.x + s2.y * s2.y;
					myFloat dens_s1 = s1.x * s1.x + s1.y * s1.y;
					myFloat dens_s0 = s0.x * s0.x + s0.y * s0.y;
					myFloat dens_s_1 = s_1.x * s_1.x + s_1.y * s_1.y;
					myFloat dens_s_2 = s_2.x * s_2.x + s_2.y * s_2.y;
					myFloat dens = dens_s2 + dens_s1 + dens_s0 + dens_s_1 + dens_s_2;

					myFloat s0_i = 0;
					if (DENSITY_THRESHOLD < dens)
						s0_i = (myFloat)(s0.y / sqrt(dens));
					swapEnd(s0_i);
					file.write((char*)&s0_i, sizeof(myFloat));
				}
			}
		}
	}

	file << std::endl << "SCALARS r_m=-1 myFloat 1" << std::endl;
	file << "LOOKUP_TABLE default" << std::endl;

	for (uint z = 0; z < dzsize; ++z)
	{
		for (uint x = 0; x < dxsize; ++x)
		{
			for (uint y = 0; y < dysize; ++y)
			{
				const uint idx = z * dxsize * dysize + y * dxsize + x;
				for (uint dualNode = 0; dualNode < VALUES_IN_BLOCK; ++dualNode)
				{
					myFloat2 s2 = pPsi[idx].values[dualNode].s2;
					myFloat2 s1 = pPsi[idx].values[dualNode].s1;
					myFloat2 s0 = pPsi[idx].values[dualNode].s0;
					myFloat2 s_1 = pPsi[idx].values[dualNode].s_1;
					myFloat2 s_2 = pPsi[idx].values[dualNode].s_2;
#if BASIS == X_QUANTIZED
					myFloat c = sqrt(6) * 0.25;
					myFloat2 x_s2  = 0.25 * s2 + 0.5 * s1 +   c * s0 + 0.5 * s_1 + 0.25 * s_2;
					myFloat2 x_s1  = -0.5 * s2 - 0.5 * s1            + 0.5 * s_1 +  0.5 * s_2;
					myFloat2 x_s0  =    c * s2            - 0.5 * s0             +    c * s_2;
					myFloat2 x_s_1 = -0.5 * s2 + 0.5 * s1            - 0.5 * s_1 +  0.5 * s_2;
					myFloat2 x_s_2 = 0.25 * s2 - 0.5 * s1 +   c * s0 - 0.5 * s_1 + 0.25 * s_2;

					s2 = x_s2;
					s1 = x_s1;
					s0 = x_s0;
					s_1 = x_s_1;
					s_2 = x_s_2;
#elif BASIS == Y_QUANTIZED
					myFloat c = sqrt(6) * 0.25;
					myFloat2 im = { 0, 1 };
					myFloat2 y_s2  =      0.25 * s2 - im * 0.5 * s1 -   c * s0 + im * 0.5 * s_1 +     0.25 * s_2;
					myFloat2 y_s1  = -im * 0.5 * s2 -      0.5 * s1            -      0.5 * s_1 + im * 0.5 * s_2;
					myFloat2 y_s0  =        -c * s2                 - 0.5 * s0                  -        c * s_2;
					myFloat2 y_s_1 =  im * 0.5 * s2 -      0.5 * s1            -      0.5 * s_1 - im * 0.5 * s_2;
					myFloat2 y_s_2 =      0.25 * s2 + im * 0.5 * s1 -   c * s0 - im * 0.5 * s_1 +     0.25 * s_2;

					s2 = y_s2;
					s1 = y_s1;
					s0 = y_s0;
					s_1 = y_s_1;
					s_2 = y_s_2;
#endif

					myFloat dens_s2 = s2.x * s2.x + s2.y * s2.y;
					myFloat dens_s1 = s1.x * s1.x + s1.y * s1.y;
					myFloat dens_s0 = s0.x * s0.x + s0.y * s0.y;
					myFloat dens_s_1 = s_1.x * s_1.x + s_1.y * s_1.y;
					myFloat dens_s_2 = s_2.x * s_2.x + s_2.y * s_2.y;
					myFloat dens = dens_s2 + dens_s1 + dens_s0 + dens_s_1 + dens_s_2;

					myFloat s_1_r = 0;
					if (DENSITY_THRESHOLD < dens)
						s_1_r = (myFloat)(s_1.x / sqrt(dens));
					swapEnd(s_1_r);
					file.write((char*)&s_1_r, sizeof(myFloat));
				}
			}
		}
	}
	file << std::endl << "SCALARS i_m=-1 myFloat 1" << std::endl;
	file << "LOOKUP_TABLE default" << std::endl;

	for (uint z = 0; z < dzsize; ++z)
	{
		for (uint x = 0; x < dxsize; ++x)
		{
			for (uint y = 0; y < dysize; ++y)
			{
				const uint idx = z * dxsize * dysize + y * dxsize + x;
				for (uint dualNode = 0; dualNode < VALUES_IN_BLOCK; ++dualNode)
				{
					myFloat2 s2 = pPsi[idx].values[dualNode].s2;
					myFloat2 s1 = pPsi[idx].values[dualNode].s1;
					myFloat2 s0 = pPsi[idx].values[dualNode].s0;
					myFloat2 s_1 = pPsi[idx].values[dualNode].s_1;
					myFloat2 s_2 = pPsi[idx].values[dualNode].s_2;
#if BASIS == X_QUANTIZED
					myFloat c = sqrt(6) * 0.25;
					myFloat2 x_s2  = 0.25 * s2 + 0.5 * s1 +   c * s0 + 0.5 * s_1 + 0.25 * s_2;
					myFloat2 x_s1  = -0.5 * s2 - 0.5 * s1            + 0.5 * s_1 +  0.5 * s_2;
					myFloat2 x_s0  =    c * s2            - 0.5 * s0             +    c * s_2;
					myFloat2 x_s_1 = -0.5 * s2 + 0.5 * s1            - 0.5 * s_1 +  0.5 * s_2;
					myFloat2 x_s_2 = 0.25 * s2 - 0.5 * s1 +   c * s0 - 0.5 * s_1 + 0.25 * s_2;

					s2 = x_s2;
					s1 = x_s1;
					s0 = x_s0;
					s_1 = x_s_1;
					s_2 = x_s_2;
#elif BASIS == Y_QUANTIZED
					myFloat c = sqrt(6) * 0.25;
					myFloat2 im = { 0, 1 };
					myFloat2 y_s2  =      0.25 * s2 - im * 0.5 * s1 -   c * s0 + im * 0.5 * s_1 +     0.25 * s_2;
					myFloat2 y_s1  = -im * 0.5 * s2 -      0.5 * s1            -      0.5 * s_1 + im * 0.5 * s_2;
					myFloat2 y_s0  =        -c * s2                 - 0.5 * s0                  -        c * s_2;
					myFloat2 y_s_1 =  im * 0.5 * s2 -      0.5 * s1            -      0.5 * s_1 - im * 0.5 * s_2;
					myFloat2 y_s_2 =      0.25 * s2 + im * 0.5 * s1 -   c * s0 - im * 0.5 * s_1 +     0.25 * s_2;

					s2 = y_s2;
					s1 = y_s1;
					s0 = y_s0;
					s_1 = y_s_1;
					s_2 = y_s_2;
#endif

					myFloat dens_s2 = s2.x * s2.x + s2.y * s2.y;
					myFloat dens_s1 = s1.x * s1.x + s1.y * s1.y;
					myFloat dens_s0 = s0.x * s0.x + s0.y * s0.y;
					myFloat dens_s_1 = s_1.x * s_1.x + s_1.y * s_1.y;
					myFloat dens_s_2 = s_2.x * s_2.x + s_2.y * s_2.y;
					myFloat dens = dens_s2 + dens_s1 + dens_s0 + dens_s_1 + dens_s_2;

					myFloat s_1_i = 0;
					if (DENSITY_THRESHOLD < dens)
						s_1_i = (myFloat)(s_1.y / sqrt(dens));
					swapEnd(s_1_i);
					file.write((char*)&s_1_i, sizeof(myFloat));
				}
			}
		}
	}

	file << std::endl << "SCALARS r_m=-2 myFloat 1" << std::endl;
	file << "LOOKUP_TABLE default" << std::endl;

	for (uint z = 0; z < dzsize; ++z)
	{
		for (uint x = 0; x < dxsize; ++x)
		{
			for (uint y = 0; y < dysize; ++y)
			{
				const uint idx = z * dxsize * dysize + y * dxsize + x;
				for (uint dualNode = 0; dualNode < VALUES_IN_BLOCK; ++dualNode)
				{
					myFloat2 s2 = pPsi[idx].values[dualNode].s2;
					myFloat2 s1 = pPsi[idx].values[dualNode].s1;
					myFloat2 s0 = pPsi[idx].values[dualNode].s0;
					myFloat2 s_1 = pPsi[idx].values[dualNode].s_1;
					myFloat2 s_2 = pPsi[idx].values[dualNode].s_2;
#if BASIS == X_QUANTIZED
					myFloat c = sqrt(6) * 0.25;
					myFloat2 x_s2  = 0.25 * s2 + 0.5 * s1 +   c * s0 + 0.5 * s_1 + 0.25 * s_2;
					myFloat2 x_s1  = -0.5 * s2 - 0.5 * s1            + 0.5 * s_1 +  0.5 * s_2;
					myFloat2 x_s0  =    c * s2            - 0.5 * s0             +    c * s_2;
					myFloat2 x_s_1 = -0.5 * s2 + 0.5 * s1            - 0.5 * s_1 +  0.5 * s_2;
					myFloat2 x_s_2 = 0.25 * s2 - 0.5 * s1 +   c * s0 - 0.5 * s_1 + 0.25 * s_2;

					s2 = x_s2;
					s1 = x_s1;
					s0 = x_s0;
					s_1 = x_s_1;
					s_2 = x_s_2;
#elif BASIS == Y_QUANTIZED
					myFloat c = sqrt(6) * 0.25;
					myFloat2 im = { 0, 1 };
					myFloat2 y_s2  =      0.25 * s2 - im * 0.5 * s1 -   c * s0 + im * 0.5 * s_1 +     0.25 * s_2;
					myFloat2 y_s1  = -im * 0.5 * s2 -      0.5 * s1            -      0.5 * s_1 + im * 0.5 * s_2;
					myFloat2 y_s0  =        -c * s2                 - 0.5 * s0                  -        c * s_2;
					myFloat2 y_s_1 =  im * 0.5 * s2 -      0.5 * s1            -      0.5 * s_1 - im * 0.5 * s_2;
					myFloat2 y_s_2 =      0.25 * s2 + im * 0.5 * s1 -   c * s0 - im * 0.5 * s_1 +     0.25 * s_2;

					s2 = y_s2;
					s1 = y_s1;
					s0 = y_s0;
					s_1 = y_s_1;
					s_2 = y_s_2;
#endif

					myFloat dens_s2 = s2.x * s2.x + s2.y * s2.y;
					myFloat dens_s1 = s1.x * s1.x + s1.y * s1.y;
					myFloat dens_s0 = s0.x * s0.x + s0.y * s0.y;
					myFloat dens_s_1 = s_1.x * s_1.x + s_1.y * s_1.y;
					myFloat dens_s_2 = s_2.x * s_2.x + s_2.y * s_2.y;
					myFloat dens = dens_s2 + dens_s1 + dens_s0 + dens_s_1 + dens_s_2;

					myFloat s_2_r = 0;
					if (DENSITY_THRESHOLD < dens)
						s_2_r = (myFloat)(s_2.x / sqrt(dens));
					swapEnd(s_2_r);
					file.write((char*)&s_2_r, sizeof(myFloat));
				}
			}
		}
	}
	file << std::endl << "SCALARS i_m=-2 myFloat 1" << std::endl;
	file << "LOOKUP_TABLE default" << std::endl;

	for (uint z = 0; z < dzsize; ++z)
	{
		for (uint x = 0; x < dxsize; ++x)
		{
			for (uint y = 0; y < dysize; ++y)
			{
				const uint idx = z * dxsize * dysize + y * dxsize + x;
				for (uint dualNode = 0; dualNode < VALUES_IN_BLOCK; ++dualNode)
				{
					myFloat2 s2 = pPsi[idx].values[dualNode].s2;
					myFloat2 s1 = pPsi[idx].values[dualNode].s1;
					myFloat2 s0 = pPsi[idx].values[dualNode].s0;
					myFloat2 s_1 = pPsi[idx].values[dualNode].s_1;
					myFloat2 s_2 = pPsi[idx].values[dualNode].s_2;
#if BASIS == X_QUANTIZED
					myFloat c = sqrt(6) * 0.25;
					myFloat2 x_s2  = 0.25 * s2 + 0.5 * s1 +   c * s0 + 0.5 * s_1 + 0.25 * s_2;
					myFloat2 x_s1  = -0.5 * s2 - 0.5 * s1            + 0.5 * s_1 +  0.5 * s_2;
					myFloat2 x_s0  =    c * s2            - 0.5 * s0             +    c * s_2;
					myFloat2 x_s_1 = -0.5 * s2 + 0.5 * s1            - 0.5 * s_1 +  0.5 * s_2;
					myFloat2 x_s_2 = 0.25 * s2 - 0.5 * s1 +   c * s0 - 0.5 * s_1 + 0.25 * s_2;

					s2 = x_s2;
					s1 = x_s1;
					s0 = x_s0;
					s_1 = x_s_1;
					s_2 = x_s_2;
#elif BASIS == Y_QUANTIZED
					myFloat c = sqrt(6) * 0.25;
					myFloat2 im = { 0, 1 };
					myFloat2 y_s2  =      0.25 * s2 - im * 0.5 * s1 -   c * s0 + im * 0.5 * s_1 +     0.25 * s_2;
					myFloat2 y_s1  = -im * 0.5 * s2 -      0.5 * s1            -      0.5 * s_1 + im * 0.5 * s_2;
					myFloat2 y_s0  =        -c * s2                 - 0.5 * s0                  -        c * s_2;
					myFloat2 y_s_1 =  im * 0.5 * s2 -      0.5 * s1            -      0.5 * s_1 - im * 0.5 * s_2;
					myFloat2 y_s_2 =      0.25 * s2 + im * 0.5 * s1 -   c * s0 - im * 0.5 * s_1 +     0.25 * s_2;

					s2 = y_s2;
					s1 = y_s1;
					s0 = y_s0;
					s_1 = y_s_1;
					s_2 = y_s_2;
#endif

					myFloat dens_s2 = s2.x * s2.x + s2.y * s2.y;
					myFloat dens_s1 = s1.x * s1.x + s1.y * s1.y;
					myFloat dens_s0 = s0.x * s0.x + s0.y * s0.y;
					myFloat dens_s_1 = s_1.x * s_1.x + s_1.y * s_1.y;
					myFloat dens_s_2 = s_2.x * s_2.x + s_2.y * s_2.y;
					myFloat dens = dens_s2 + dens_s1 + dens_s0 + dens_s_1 + dens_s_2;

					myFloat s_2_i = 0;
					if (DENSITY_THRESHOLD < dens)
						s_2_i = (myFloat)(s_2.y / sqrt(dens));
					swapEnd(s_2_i);
					file.write((char*)&s_2_i, sizeof(myFloat));
				}
			}
		}
	}

	//file << "VERTICES " << pointCount << " " << pointCount << std::endl;
	//for (int i = 0; i < pointCount; ++i)
	//{
	//	int swapped = i;
	//	swapEnd(swapped);
	//	file.write((char*)&swapped, sizeof(int));
	//}

	file << std::endl;
	file.close();
}

myFloat3 centerOfMass(BlockPsis* h_evenPsi, size_t bsize, size_t dxsize, size_t dysize, size_t dzsize, myFloat block_scale, myFloat3 p0)
{
	myFloat3 com{};

	myFloat totDens = 0;

	for (uint z = 0; z < dzsize; ++z)
	{
		for (uint x = 0; x < dxsize; ++x)
		{
			for (uint y = 0; y < dysize; ++y)
			{
				const uint idx = z * dxsize * dysize + y * dxsize + x;
				for (uint dualNode = 0; dualNode < VALUES_IN_BLOCK; ++dualNode)
				{
					myFloat3 localPos = getLocalPos(dualNode);
					myFloat3 globalPos = { p0.x + block_scale * ((x - 1.0) * BLOCK_WIDTH_X + localPos.x),
										  p0.y + block_scale * ((y - 1.0) * BLOCK_WIDTH_Y + localPos.y),
										  p0.z + block_scale * ((z - 1.0) * BLOCK_WIDTH_Z + localPos.z) };

					myFloat normSq_s1 = h_evenPsi[idx].values[dualNode].s1.x * h_evenPsi[idx].values[dualNode].s1.x + h_evenPsi[idx].values[dualNode].s1.y * h_evenPsi[idx].values[dualNode].s1.y;
					myFloat normSq_s0 = h_evenPsi[idx].values[dualNode].s0.x * h_evenPsi[idx].values[dualNode].s0.x + h_evenPsi[idx].values[dualNode].s0.y * h_evenPsi[idx].values[dualNode].s0.y;
					myFloat normSq_s_1 = h_evenPsi[idx].values[dualNode].s_1.x * h_evenPsi[idx].values[dualNode].s_1.x + h_evenPsi[idx].values[dualNode].s_1.y * h_evenPsi[idx].values[dualNode].s_1.y;
					myFloat density = normSq_s1 + normSq_s0 + normSq_s_1;

					com += density * globalPos;
					totDens += density;
				}
			}
		}
	}

	return com 	/ totDens;
}

#endif // UTILS