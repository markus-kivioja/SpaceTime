#ifndef UTILS
#define UTILS

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
__host__ __device__ __inline__ myFloat2 operator-(myFloat2 a)
{
	return { -a.x, -a.y };
}
__host__ __device__ __inline__ myFloat3 operator-(myFloat3 a, myFloat3 b)
{
	return { a.x - b.x, a.y - b.y, a.z - b.z };
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
__host__ __device__ __inline__ myFloat2 operator*(myFloat2 a, myFloat2 b) // Complex number multiplication
{
	return { a.x * b.x - a.y * b.y, a.y * b.x + a.x * b.y };
}

struct Complex3Vec
{
	myFloat2 s1;
	myFloat2 s0;
	myFloat2 s_1;
};

struct BlockPsis
{
	Complex3Vec values[VALUES_IN_BLOCK];
};

struct BlockEdges
{
	Complex3Vec values[EDGES_IN_BLOCK];
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

std::string toString(const myFloat value)
{
	std::ostringstream out;
	out.precision(18);
	out << std::fixed << value;
	return out.str();
};

void drawIandR(const std::string& folder, BlockPsis* h_evenPsi, size_t dxsize, size_t dysize, size_t dzsize, myFloat t, MagFields Bs, const myFloat3 p0, myFloat block_scale)
{
	const int SIZE = 2;
	const myFloat INTENSITY = 1;
	const int width = dxsize * SIZE, height = dysize * SIZE, depth = dzsize * SIZE;
	Picture pic1(width * 3, height * 2);

	// XZ-plane
	for (uint k = 0; k < depth; ++k)
	{
		for (uint i = 0; i < width; i++)
		{
			myFloat2 norm_s1 = { 0, 0 };
			myFloat2 norm_s0 = { 0, 0 };
			myFloat2 norm_s_1 = { 0, 0 };
			for (uint j = 0; j < height; j++)
			{
				const uint idx = (k / SIZE) * dxsize * dysize + (j / SIZE) * dxsize + i / SIZE;
				for (uint dualNode = 0; dualNode < VALUES_IN_BLOCK; ++dualNode)
				{
					norm_s1 += {h_evenPsi[idx].values[dualNode].s1.x* h_evenPsi[idx].values[dualNode].s1.x, h_evenPsi[idx].values[dualNode].s1.y* h_evenPsi[idx].values[dualNode].s1.y  };
					norm_s0 += {h_evenPsi[idx].values[dualNode].s0.x* h_evenPsi[idx].values[dualNode].s0.x, h_evenPsi[idx].values[dualNode].s0.y* h_evenPsi[idx].values[dualNode].s0.y  };
					norm_s_1 += {h_evenPsi[idx].values[dualNode].s_1.x* h_evenPsi[idx].values[dualNode].s_1.x, h_evenPsi[idx].values[dualNode].s_1.y* h_evenPsi[idx].values[dualNode].s_1.y};
				}
			}
			{
				const myFloat2 s1 = INTENSITY * norm_s1;
				const myFloat2 s0 = INTENSITY * norm_s0;
				const myFloat2 s_1 = INTENSITY * norm_s_1;
				pic1.setColor(i, k, Vector4(s1.x, s1.y, 0.0, 1.0));
				pic1.setColor(width + i, k, Vector4(s0.x, s0.y, 0.0, 1.0));
				pic1.setColor(2 * width + i, k, Vector4(s_1.x, s_1.y, 0.0, 1.0));
			}
		}
	}

	// XY-plane
	for (uint j = 0; j < height; j++)
	{
		for (uint i = 0; i < width; i++)
		{
			myFloat2 norm_s1 = { 0, 0 };
			myFloat2 norm_s0 = { 0, 0 };
			myFloat2 norm_s_1 = { 0, 0 };
			for (uint k = 0; k < depth; ++k)
			{
				const uint idx = (k / SIZE) * dxsize * dysize + (j / SIZE) * dxsize + i / SIZE;
				for (uint dualNode = 0; dualNode < VALUES_IN_BLOCK; ++dualNode)
				{
					norm_s1 += {h_evenPsi[idx].values[dualNode].s1.x* h_evenPsi[idx].values[dualNode].s1.x, h_evenPsi[idx].values[dualNode].s1.y* h_evenPsi[idx].values[dualNode].s1.y  };
					norm_s0 += {h_evenPsi[idx].values[dualNode].s0.x* h_evenPsi[idx].values[dualNode].s0.x, h_evenPsi[idx].values[dualNode].s0.y* h_evenPsi[idx].values[dualNode].s0.y  };
					norm_s_1 += {h_evenPsi[idx].values[dualNode].s_1.x* h_evenPsi[idx].values[dualNode].s_1.x, h_evenPsi[idx].values[dualNode].s_1.y* h_evenPsi[idx].values[dualNode].s_1.y};
				}
			}
			{
				const myFloat2 s1 = INTENSITY * norm_s1;
				const myFloat2 s0 = INTENSITY * norm_s0;
				const myFloat2 s_1 = INTENSITY * norm_s_1;

				pic1.setColor(i, height + j, Vector4(s1.x, s1.y, 0.0, 1.0));
				pic1.setColor(width + i, height + j, Vector4(s0.x, s0.y, 0.0, 1.0));
				pic1.setColor(2 * width + i, height + j, Vector4(s_1.x, s_1.y, 0.0, 1.0));
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

void drawDensity(const std::string& filePrefix, BlockPsis* h_evenPsi, size_t dxsize, size_t dysize, size_t dzsize, myFloat t, const std::string& folder)
{
	const int SIZE = 2;
	myFloat INTENSITY = 1.0;

	const int width = dxsize * SIZE, height = dysize * SIZE, depth = dzsize * SIZE;
	Picture pic1(width * 3, height * 2);

	myFloat maxVal = 0;
	// XZ-plane
	for (uint k = 0; k < depth; ++k)
	{
		for (uint i = 0; i < width; i++)
		{
			myFloat norm_s1 = 0;
			myFloat norm_s0 = 0;
			myFloat norm_s_1 = 0;
			for (uint j = 0; j < height; j++)
			{
				const uint idx = (k / SIZE) * dxsize * dysize + (j / SIZE) * dxsize + i / SIZE;
				for (uint dualNode = 0; dualNode < VALUES_IN_BLOCK; ++dualNode)
				{
					myFloat2 s1 = h_evenPsi[idx].values[dualNode].s1;
					myFloat2 s0 = h_evenPsi[idx].values[dualNode].s0;
					myFloat2 s_1 = h_evenPsi[idx].values[dualNode].s_1;

#if BASIS == X_QUANTIZED
					myFloat2 x_s1 = 0.5 * (s1 + s_1) - s0 / sqrt(2);
					myFloat2 x_s0 = (s1 - s_1) / sqrt(2);
					myFloat2 x_s_1 = 0.5 * (s_1 + s1) + s0 / sqrt(2);

					s1 = x_s1;
					s0 = x_s0;
					s_1 = x_s_1;
#elif BASIS == Y_QUANTIZED
					myFloat2 y_s1 = 0.5 * (s1 - s_1) + s0 * myFloat2{ 0, -1 } / sqrt(2);
					myFloat2 y_s0 = (s1 + s_1) * myFloat2{ 0, -1 } / sqrt(2);
					myFloat2 y_s_1 = 0.5 * (s_1 - s1) + s0 * myFloat2{ 0, -1 } / sqrt(2);

					s1 = y_s1;
					s0 = y_s0;
					s_1 = y_s_1;
#endif

					norm_s1 += s1.x * s1.x + s1.y * s1.y;
					norm_s0 += s0.x * s0.x + s0.y * s0.y;
					norm_s_1 += s_1.x * s_1.x + s_1.y * s_1.y;
				}
			}
			maxVal = std::max(maxVal, std::max(norm_s1, std::max(norm_s0, norm_s_1)));
		}
	}
	INTENSITY = 1.0 / maxVal;

	// XZ-plane
	for (uint k = 0; k < depth; ++k)
	{
		for (uint i = 0; i < width; i++)
		{
			myFloat norm_s1 = 0;
			myFloat norm_s0 = 0;
			myFloat norm_s_1 = 0;
			for (uint j = 0; j < height; j++)
			{
				const uint idx = (k / SIZE) * dxsize * dysize + (j / SIZE) * dxsize + i / SIZE;
				for (uint dualNode = 0; dualNode < VALUES_IN_BLOCK; ++dualNode)
				{
					myFloat2 s1 = h_evenPsi[idx].values[dualNode].s1;
					myFloat2 s0 = h_evenPsi[idx].values[dualNode].s0;
					myFloat2 s_1 = h_evenPsi[idx].values[dualNode].s_1;

#if BASIS == X_QUANTIZED
					myFloat2 x_s1 = 0.5 * (s1 + s_1) - s0 / sqrt(2);
					myFloat2 x_s0 = (s1 - s_1) / sqrt(2);
					myFloat2 x_s_1 = 0.5 * (s_1 + s1) + s0 / sqrt(2);

					s1 = x_s1;
					s0 = x_s0;
					s_1 = x_s_1;
#elif BASIS == Y_QUANTIZED
					myFloat2 y_s1 = 0.5 * (s1 - s_1) + s0 * myFloat2{ 0, -1 } / sqrt(2);
					myFloat2 y_s0 = (s1 + s_1) * myFloat2{ 0, -1 } / sqrt(2);
					myFloat2 y_s_1 = 0.5 * (s_1 - s1) + s0 * myFloat2{ 0, -1 } / sqrt(2);

					s1 = y_s1;
					s0 = y_s0;
					s_1 = y_s_1;
#endif

					norm_s1 += s1.x * s1.x + s1.y * s1.y;
					norm_s0 += s0.x * s0.x + s0.y * s0.y;
					norm_s_1 += s_1.x * s_1.x + s_1.y * s_1.y;
				}
			}

			const myFloat s1 = INTENSITY * norm_s1;
			const myFloat s0 = INTENSITY * norm_s0;
			const myFloat s_1 = INTENSITY * norm_s_1;

			int uv_x = (width - i - 1);
			pic1.setColor(uv_x, k, Vector4(s1, s1, s1, 1.0));
			pic1.setColor(width + uv_x, k, Vector4(s0, s0, s0, 1.0));
			pic1.setColor(2 * width + uv_x, k, Vector4(s_1, s_1, s_1, 1.0));
		}
	}

	// XY-plane
	maxVal = 0;
	for (uint j = 0; j < height; j++)
	{
		for (uint i = 0; i < width; i++)
		{
			myFloat norm_s1 = 0;
			myFloat norm_s0 = 0;
			myFloat norm_s_1 = 0;
			for (uint k = 0; k < depth; ++k)
			{
				const uint idx = (k / SIZE) * dxsize * dysize + (j / SIZE) * dxsize + i / SIZE;
				for (uint dualNode = 0; dualNode < VALUES_IN_BLOCK; ++dualNode)
				{
					myFloat2 s1 = h_evenPsi[idx].values[dualNode].s1;
					myFloat2 s0 = h_evenPsi[idx].values[dualNode].s0;
					myFloat2 s_1 = h_evenPsi[idx].values[dualNode].s_1;

#if BASIS == X_QUANTIZED
					myFloat2 x_s1 = 0.5 * (s1 + s_1) - s0 / sqrt(2);
					myFloat2 x_s0 = (s1 - s_1) / sqrt(2);
					myFloat2 x_s_1 = 0.5 * (s_1 + s1) + s0 / sqrt(2);

					s1 = x_s1;
					s0 = x_s0;
					s_1 = x_s_1;
#elif BASIS == Y_QUANTIZED
					myFloat2 y_s1 = 0.5 * (s1 - s_1) + s0 * myFloat2{ 0, -1 } / sqrt(2);
					myFloat2 y_s0 = (s1 + s_1) * myFloat2{ 0, -1 } / sqrt(2);
					myFloat2 y_s_1 = 0.5 * (s_1 - s1) + s0 * myFloat2{ 0, -1 } / sqrt(2);

					s1 = y_s1;
					s0 = y_s0;
					s_1 = y_s_1;
#endif

					norm_s1 += s1.x * s1.x + s1.y * s1.y;
					norm_s0 += s0.x * s0.x + s0.y * s0.y;
					norm_s_1 += s_1.x * s_1.x + s_1.y * s_1.y;
				}
			}
			maxVal = std::max(maxVal, std::max(norm_s1, std::max(norm_s0, norm_s_1)));
		}
	}
	INTENSITY = 1.0 / maxVal;
	// XY-plane
	for (uint j = 0; j < height; j++)
	{
		for (uint i = 0; i < width; i++)
		{
			myFloat norm_s1 = 0;
			myFloat norm_s0 = 0;
			myFloat norm_s_1 = 0;
			for (uint k = 0; k < depth; ++k)
			{
				const uint idx = (k / SIZE) * dxsize * dysize + (j / SIZE) * dxsize + i / SIZE;
				for (uint dualNode = 0; dualNode < VALUES_IN_BLOCK; ++dualNode)
				{
					myFloat2 s1 = h_evenPsi[idx].values[dualNode].s1;
					myFloat2 s0 = h_evenPsi[idx].values[dualNode].s0;
					myFloat2 s_1 = h_evenPsi[idx].values[dualNode].s_1;

#if BASIS == X_QUANTIZED
					myFloat2 x_s1 = 0.5 * (s1 + s_1) - s0 / sqrt(2);
					myFloat2 x_s0 = (s1 - s_1) / sqrt(2);
					myFloat2 x_s_1 = 0.5 * (s_1 + s1) + s0 / sqrt(2);

					s1 = x_s1;
					s0 = x_s0;
					s_1 = x_s_1;
#elif BASIS == Y_QUANTIZED
					myFloat2 y_s1 = 0.5 * (s1 - s_1) + s0 * myFloat2{ 0, -1 } / sqrt(2);
					myFloat2 y_s0 = (s1 + s_1) * myFloat2{ 0, -1 } / sqrt(2);
					myFloat2 y_s_1 = 0.5 * (s_1 - s1) + s0 * myFloat2{ 0, -1 } / sqrt(2);

					s1 = y_s1;
					s0 = y_s0;
					s_1 = y_s_1;
#endif

					norm_s1 += s1.x * s1.x + s1.y * s1.y;
					norm_s0 += s0.x * s0.x + s0.y * s0.y;
					norm_s_1 += s_1.x * s_1.x + s_1.y * s_1.y;
				}
			}

			const myFloat s1 = INTENSITY * norm_s1;
			const myFloat s0 = INTENSITY * norm_s0;
			const myFloat s_1 = INTENSITY * norm_s_1;

			int uv_x = (width - i - 1);
			int uv_y = (height - j - 1);
			pic1.setColor(uv_x, height + uv_y, Vector4(s1, s1, s1, 1.0));
			pic1.setColor(width + uv_x, height + uv_y, Vector4(s0, s0, s0, 1.0));
			pic1.setColor(2 * width + uv_x, height + uv_y, Vector4(s_1, s_1, s_1, 1.0));
		}
	}

	for (int x = 0; x < width * 3; ++x)
	{
		pic1.setColor(x, height, Vector4(0.5, 0.5, 0.5, 1.0));
	}
	for (int y = 0; y < height * 2; ++y)
	{
		pic1.setColor(width, y, Vector4(0.5, 0.5, 0.5, 1.0));
		pic1.setColor(2 * width, y, Vector4(0.5, 0.5, 0.5, 1.0));
	}

	pic1.save(folder + "/" + filePrefix + "_" + toString(t) + "ms.bmp", false);
}

void drawDensityRI(const std::string& filePrefix, BlockPsis* h_evenPsi, size_t dxsize, size_t dysize, size_t dzsize, myFloat t, const std::string& folder)
{
	const int SIZE = 2;
	myFloat INTENSITY = 1.0;

	const int width = dxsize * SIZE, height = dysize * SIZE, depth = dzsize * SIZE;
	Picture pic1(width * 3, height * 2);

	myFloat maxVal = 0;
	// XZ-plane
	for (uint k = 0; k < depth; ++k)
	{
		for (uint i = 0; i < width; i++)
		{
			myFloat norm_s1 = 0;
			myFloat norm_s0 = 0;
			myFloat norm_s_1 = 0;
			for (uint j = 0; j < height; j++)
			{
				const uint idx = (k / SIZE) * dxsize * dysize + (j / SIZE) * dxsize + i / SIZE;
				for (uint dualNode = 0; dualNode < VALUES_IN_BLOCK; ++dualNode)
				{
					myFloat2 s1 = h_evenPsi[idx].values[dualNode].s1;
					myFloat2 s0 = h_evenPsi[idx].values[dualNode].s0;
					myFloat2 s_1 = h_evenPsi[idx].values[dualNode].s_1;

#if BASIS == X_QUANTIZED
					myFloat2 x_s1 = 0.5 * (s1 + s_1) - s0 / sqrt(2);
					myFloat2 x_s0 = (s1 - s_1) / sqrt(2);
					myFloat2 x_s_1 = 0.5 * (s_1 + s1) + s0 / sqrt(2);

					s1 = x_s1;
					s0 = x_s0;
					s_1 = x_s_1;
#elif BASIS == Y_QUANTIZED
					myFloat2 y_s1 = 0.5 * (s1 - s_1) + s0 * myFloat2{ 0, -1 } / sqrt(2);
					myFloat2 y_s0 = (s1 + s_1) * myFloat2{ 0, -1 } / sqrt(2);
					myFloat2 y_s_1 = 0.5 * (s_1 - s1) + s0 * myFloat2{ 0, -1 } / sqrt(2);

					s1 = y_s1;
					s0 = y_s0;
					s_1 = y_s_1;
#endif

					norm_s1 += s1.x * s1.x + s1.y * s1.y;
					norm_s0 += s0.x * s0.x + s0.y * s0.y;
					norm_s_1 += s_1.x * s_1.x + s_1.y * s_1.y;
				}
			}
			maxVal = std::max(maxVal, std::max(norm_s1, std::max(norm_s0, norm_s_1)));
		}
	}
	INTENSITY = 2.0 / maxVal;

	// XZ-plane
	for (uint k = 0; k < depth; ++k)
	{
		for (uint i = 0; i < width; i++)
		{
			myFloat2 norm_s1 = myFloat2{ 0, 0 };
			myFloat2 norm_s0 = myFloat2{ 0, 0 };
			myFloat2 norm_s_1 = myFloat2{ 0, 0 };
			for (uint j = 0; j < height; j++)
			{
				const uint idx = (k / SIZE) * dxsize * dysize + (j / SIZE) * dxsize + i / SIZE;
				for (uint dualNode = 0; dualNode < VALUES_IN_BLOCK; ++dualNode)
				{
					myFloat2 s1 = h_evenPsi[idx].values[dualNode].s1;
					myFloat2 s0 = h_evenPsi[idx].values[dualNode].s0;
					myFloat2 s_1 = h_evenPsi[idx].values[dualNode].s_1;

#if BASIS == X_QUANTIZED
					myFloat2 x_s1 = 0.5 * (s1 + s_1) - s0 / sqrt(2);
					myFloat2 x_s0 = (s1 - s_1) / sqrt(2);
					myFloat2 x_s_1 = 0.5 * (s_1 + s1) + s0 / sqrt(2);

					s1 = x_s1;
					s0 = x_s0;
					s_1 = x_s_1;
#elif BASIS == Y_QUANTIZED
					myFloat2 y_s1 = 0.5 * (s1 - s_1) + s0 * myFloat2{ 0, -1 } / sqrt(2);
					myFloat2 y_s0 = (s1 + s_1) * myFloat2{ 0, -1 } / sqrt(2);
					myFloat2 y_s_1 = 0.5 * (s_1 - s1) + s0 * myFloat2{ 0, -1 } / sqrt(2);

					s1 = y_s1;
					s0 = y_s0;
					s_1 = y_s_1;
#endif

					norm_s1 += myFloat2{ s1.x * s1.x, s1.y * s1.y };
					norm_s0 += myFloat2{ s0.x * s0.x, s0.y * s0.y };
					norm_s_1 += myFloat2{ s_1.x * s_1.x, s_1.y * s_1.y };
				}
			}

			const myFloat2 s1 = INTENSITY * norm_s1;
			const myFloat2 s0 = INTENSITY * norm_s0;
			const myFloat2 s_1 = INTENSITY * norm_s_1;

			const myFloat s1Mag = sqrt((conj(norm_s1) * norm_s1).x);
			const myFloat s0Mag = sqrt((conj(norm_s0) * norm_s0).x);
			const myFloat s_1Mag = sqrt((conj(norm_s_1) * norm_s_1).x);

			pic1.setColor(i, k, Vector4(s1.x / s1Mag, s1.y / s1Mag, 0.0, 1.0));
			pic1.setColor(width + i, k, Vector4(s0.x / s0Mag, s0.y / s0Mag, 0.0, 1.0));
			pic1.setColor(2 * width + i, k, Vector4(s_1.x / s_1Mag, s_1.y / s_1Mag, 0.0, 1.0));
		}
	}

	// XY-plane
	maxVal = 0;
	for (uint j = 0; j < height; j++)
	{
		for (uint i = 0; i < width; i++)
		{
			myFloat norm_s1 = 0;
			myFloat norm_s0 = 0;
			myFloat norm_s_1 = 0;
			for (uint k = 0; k < depth; ++k)
			{
				const uint idx = (k / SIZE) * dxsize * dysize + (j / SIZE) * dxsize + i / SIZE;
				for (uint dualNode = 0; dualNode < VALUES_IN_BLOCK; ++dualNode)
				{
					myFloat2 s1 = h_evenPsi[idx].values[dualNode].s1;
					myFloat2 s0 = h_evenPsi[idx].values[dualNode].s0;
					myFloat2 s_1 = h_evenPsi[idx].values[dualNode].s_1;

#if BASIS == X_QUANTIZED
					myFloat2 x_s1 = 0.5 * (s1 + s_1) - s0 / sqrt(2);
					myFloat2 x_s0 = (s1 - s_1) / sqrt(2);
					myFloat2 x_s_1 = 0.5 * (s_1 + s1) + s0 / sqrt(2);

					s1 = x_s1;
					s0 = x_s0;
					s_1 = x_s_1;
#elif BASIS == Y_QUANTIZED
					myFloat2 y_s1 = 0.5 * (s1 - s_1) + s0 * myFloat2{ 0, -1 } / sqrt(2);
					myFloat2 y_s0 = (s1 + s_1) * myFloat2{ 0, -1 } / sqrt(2);
					myFloat2 y_s_1 = 0.5 * (s_1 - s1) + s0 * myFloat2{ 0, -1 } / sqrt(2);

					s1 = y_s1;
					s0 = y_s0;
					s_1 = y_s_1;
#endif

					norm_s1 += s1.x * s1.x + s1.y * s1.y;
					norm_s0 += s0.x * s0.x + s0.y * s0.y;
					norm_s_1 += s_1.x * s_1.x + s_1.y * s_1.y;
				}
			}
			maxVal = std::max(maxVal, std::max(norm_s1, std::max(norm_s0, norm_s_1)));
		}
	}
	INTENSITY = 2.0 / maxVal;
	// XY-plane
	for (uint j = 0; j < height; j++)
	{
		for (uint i = 0; i < width; i++)
		{
			myFloat2 norm_s1 = myFloat2{ 0, 0 };
			myFloat2 norm_s0 = myFloat2{ 0, 0 };
			myFloat2 norm_s_1 = myFloat2{ 0, 0 };
			for (uint k = 0; k < depth; ++k)
			{
				const uint idx = (k / SIZE) * dxsize * dysize + (j / SIZE) * dxsize + i / SIZE;
				for (uint dualNode = 0; dualNode < VALUES_IN_BLOCK; ++dualNode)
				{
					myFloat2 s1 = h_evenPsi[idx].values[dualNode].s1;
					myFloat2 s0 = h_evenPsi[idx].values[dualNode].s0;
					myFloat2 s_1 = h_evenPsi[idx].values[dualNode].s_1;

#if BASIS == X_QUANTIZED
					myFloat2 x_s1 = 0.5 * (s1 + s_1) - s0 / sqrt(2);
					myFloat2 x_s0 = (s1 - s_1) / sqrt(2);
					myFloat2 x_s_1 = 0.5 * (s_1 + s1) + s0 / sqrt(2);

					s1 = x_s1;
					s0 = x_s0;
					s_1 = x_s_1;
#elif BASIS == Y_QUANTIZED
					myFloat2 y_s1 = 0.5 * (s1 - s_1) + s0 * myFloat2{ 0, -1 } / sqrt(2);
					myFloat2 y_s0 = (s1 + s_1) * myFloat2{ 0, -1 } / sqrt(2);
					myFloat2 y_s_1 = 0.5 * (s_1 - s1) + s0 * myFloat2{ 0, -1 } / sqrt(2);

					s1 = y_s1;
					s0 = y_s0;
					s_1 = y_s_1;
#endif

					norm_s1 += myFloat2{ s1.x * s1.x, s1.y * s1.y };
					norm_s0 += myFloat2{ s0.x * s0.x, s0.y * s0.y };
					norm_s_1 += myFloat2{ s_1.x * s_1.x, s_1.y * s_1.y };
				}
			}

			const myFloat2 s1 = INTENSITY * norm_s1;
			const myFloat2 s0 = INTENSITY * norm_s0;
			const myFloat2 s_1 = INTENSITY * norm_s_1;

			pic1.setColor(i, height + j, Vector4(s1.x, s1.y, 0.0, 1.0));
			pic1.setColor(width + i, height + j, Vector4(s0.x, s0.y, 0.0, 1.0));
			pic1.setColor(2 * width + i, height + j, Vector4(s_1.x, s_1.y, 0.0, 1.0));
		}
	}

	for (int x = 0; x < width * 3; ++x)
	{
		pic1.setColor(x, height, Vector4(0.5, 0.5, 0.5, 1.0));
	}
	for (int y = 0; y < height * 2; ++y)
	{
		pic1.setColor(width, y, Vector4(0.5, 0.5, 0.5, 1.0));
		pic1.setColor(2 * width, y, Vector4(0.5, 0.5, 0.5, 1.0));
	}

	pic1.save(folder + "/" + filePrefix + "_" + toString(t) + "ms.bmp", false);
}

void drawDensityRgb(const std::string& name, BlockPsis* h_evenPsi, size_t dxsize, size_t dysize, size_t dzsize, myFloat t, const std::string& folder)
{
	const int SIZE = 4;
	const myFloat INTENSITY = 1.0;

	const int width = dxsize * SIZE, height = dysize * SIZE, depth = dzsize * SIZE;
	Picture pic1(width * 2, height);

	// XZ-plane
	for (uint k = 0; k < depth; ++k)
	{
		for (uint i = 0; i < width; i++)
		{
			myFloat norm_s1 = 0;
			myFloat norm_s0 = 0;
			myFloat norm_s_1 = 0;
			for (uint j = 0; j < height; j++)
			{
				const uint idx = (k / SIZE) * dxsize * dysize + (j / SIZE) * dxsize + i / SIZE;
				for (uint dualNode = 0; dualNode < VALUES_IN_BLOCK; ++dualNode)
				{
					norm_s1 += h_evenPsi[idx].values[dualNode].s1.x * h_evenPsi[idx].values[dualNode].s1.x + h_evenPsi[idx].values[dualNode].s1.y * h_evenPsi[idx].values[dualNode].s1.y;
					norm_s0 += h_evenPsi[idx].values[dualNode].s0.x * h_evenPsi[idx].values[dualNode].s0.x + h_evenPsi[idx].values[dualNode].s0.y * h_evenPsi[idx].values[dualNode].s0.y;
					norm_s_1 += h_evenPsi[idx].values[dualNode].s_1.x * h_evenPsi[idx].values[dualNode].s_1.x + h_evenPsi[idx].values[dualNode].s_1.y * h_evenPsi[idx].values[dualNode].s_1.y;
				}
			}

			const myFloat s1 = INTENSITY * norm_s1;
			const myFloat s0 = INTENSITY * norm_s0;
			const myFloat s_1 = INTENSITY * norm_s_1;

			pic1.setColor(i, k, Vector4(s1, s_1, s0, 1.0));
		}
	}

	// XY-plane
	for (uint j = 0; j < height; j++)
	{
		for (uint i = 0; i < width; i++)
		{
			myFloat norm_s1 = 0;
			myFloat norm_s0 = 0;
			myFloat norm_s_1 = 0;
			for (uint k = 0; k < depth; ++k)
			{
				const uint idx = (k / SIZE) * dxsize * dysize + (j / SIZE) * dxsize + i / SIZE;
				for (uint dualNode = 0; dualNode < VALUES_IN_BLOCK; ++dualNode)
				{
					norm_s1 += h_evenPsi[idx].values[dualNode].s1.x * h_evenPsi[idx].values[dualNode].s1.x + h_evenPsi[idx].values[dualNode].s1.y * h_evenPsi[idx].values[dualNode].s1.y;
					norm_s0 += h_evenPsi[idx].values[dualNode].s0.x * h_evenPsi[idx].values[dualNode].s0.x + h_evenPsi[idx].values[dualNode].s0.y * h_evenPsi[idx].values[dualNode].s0.y;
					norm_s_1 += h_evenPsi[idx].values[dualNode].s_1.x * h_evenPsi[idx].values[dualNode].s_1.x + h_evenPsi[idx].values[dualNode].s_1.y * h_evenPsi[idx].values[dualNode].s_1.y;
				}
			}

			const myFloat s1 = INTENSITY * norm_s1;
			const myFloat s0 = INTENSITY * norm_s0;
			const myFloat s_1 = INTENSITY * norm_s_1;

			pic1.setColor(width + i, j, Vector4(s1, s_1, s0, 1.0));
		}
	}

	for (int y = 0; y < height; ++y)
	{
		pic1.setColor(width, y, Vector4(1.0, 1.0, 1.0, 1.0));
	}

	uint axisOffsetX = 5;
	uint axisOffsetY = 5;
	Picture xzAxis;
	Picture xyAxis;
	xzAxis.load("xz_axis.bmp");
	xyAxis.load("xy_axis.bmp");
	for (uint x = 0; x < 60; ++x)
	{
		for (uint y = 0; y < 61; ++y)
		{
			Vector4 color = xzAxis.getColor(x, y);
			pic1.setColor(axisOffsetX + x, axisOffsetY + y, color);

			color = xyAxis.getColor(x, y);
			pic1.setColor(width + axisOffsetX + x, axisOffsetY + y, color);
		}
	}

	pic1.save(folder + "/" + name + toString(t) + "ms.bmp", false);
}

void drawUtheta(const myFloat3* uPtr, const myFloat* thetaPtr, const size_t xSize, const size_t ySize, const size_t zSize, const myFloat t)
{
	const uint SIZE = 2;
	const myFloat U_INTENSITY = 15.0;
	const myFloat THETA_INTENSITY = 24.0;

	const int width = xSize * SIZE, height = ySize * SIZE, depth = zSize * SIZE;
	Picture pic1(width * 2, height * 2);

	// XZ-plane
	uint y = height / 2;
	for (uint z = 0; z < depth; z += SIZE)
	{
		for (uint x = 0; x < width; x += SIZE)
		{
			myFloat3 us[4] = {
				myFloat3{ 0, 0, 0 },
				myFloat3{ 0, 0, 0 },
				myFloat3{ 0, 0, 0 },
				myFloat3{ 0, 0, 0 }
			};
			myFloat thetas[4] = { 0, 0, 0, 0 };
			uint counts[4] = { 0, 0, 0, 0 };

			for (uint cellIdx = 0; cellIdx < VALUES_IN_BLOCK; ++cellIdx)
			{
				const uint structIdx = VALUES_IN_BLOCK * ((z / SIZE) * xSize * ySize + (y / SIZE) * xSize + (x / SIZE));
				const uint idx = structIdx + cellIdx;

				myFloat3 localPos = getLocalPos(cellIdx);
				int localX = (int)localPos.x;
				int localZ = (int)localPos.z;

				int localIdx = localZ * SIZE + localX;
				us[localIdx] += uPtr[idx];
				thetas[localIdx] += thetaPtr[idx];

				counts[localIdx]++;
			}

			for (uint i = 0; i < 4; ++i)
			{
				myFloat3 u = us[i] / counts[i];
				myFloat norm = u.x * u.x + u.y * u.y + u.z * u.z;

				u = U_INTENSITY * u;
				myFloat theta = THETA_INTENSITY * sqrt(norm) * thetas[i] / counts[i] / PI;

				pic1.setColor(x + (i % SIZE), z + (i / SIZE), Vector4(u.x, u.y, u.z, 1.0));
				pic1.setColor(width + x + (i % SIZE), z + (i / SIZE), Vector4(-theta, theta, 0.0, 1.0));
			}
		}
	}

	// XY-plane
	uint z = depth / 2;
	for (uint y = 0; y < height; y += SIZE)
	{
		for (uint x = 0; x < width; x += SIZE)
		{
			myFloat3 us[4] = {
				myFloat3{ 0, 0, 0 },
				myFloat3{ 0, 0, 0 },
				myFloat3{ 0, 0, 0 },
				myFloat3{ 0, 0, 0 }
			};
			myFloat thetas[4] = { 0, 0, 0, 0 };
			uint counts[4] = { 0, 0, 0, 0 };

			for (uint cellIdx = 0; cellIdx < VALUES_IN_BLOCK; ++cellIdx)
			{
				const uint structIdx = VALUES_IN_BLOCK * ((z / SIZE) * xSize * ySize + (y / SIZE) * xSize + (x / SIZE));
				const uint idx = structIdx + cellIdx;

				myFloat3 localPos = getLocalPos(cellIdx);
				int localX = (int)localPos.x;
				int localY = (int)localPos.y;

				int localIdx = localY * SIZE + localX;
				us[localIdx] += uPtr[idx];
				thetas[localIdx] += thetaPtr[idx];

				counts[localIdx]++;
			}

			for (uint i = 0; i < 4; ++i)
			{
				myFloat3 u = us[i] / counts[i];
				myFloat norm = u.x * u.x + u.y * u.y + u.z * u.z;

				u = U_INTENSITY * u;
				myFloat theta = THETA_INTENSITY * sqrt(norm) * thetas[i] / counts[i] / PI;

				pic1.setColor(x + (i % SIZE), height + y + (i / SIZE), Vector4(u.x, u.y, u.z, 1.0));
				pic1.setColor(width + x + (i % SIZE), height + y + (i / SIZE), Vector4(-theta, theta, 0.0, 1.0));
			}
		}
	}

	for (int x = 0; x < width * 2; ++x)
	{
		pic1.setColor(x, height, Vector4(0.5, 0.5, 0.5, 1.0));
	}
	for (int y = 0; y < height * 2; ++y)
	{
		pic1.setColor(width, y, Vector4(0.5, 0.5, 0.5, 1.0));
	}

	pic1.save("results/u_v_theta_" + toString(t) + "ms.bmp", false);
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

void saveVolume(const std::string& namePrefix, BlockPsis* pPsi, myFloat3* pLocalAvgSpin, myFloat3* pu, myFloat* pTheta, size_t bsize, size_t dxsize, size_t dysize, size_t dzsize, uint iter, myFloat block_scale, myFloat3 p0, myFloat t, const std::string& folder)
{
	std::ofstream file;
	file.open(folder + "/" + namePrefix + std::to_string(t) + ".vtk", std::ios::out | std::ios::binary);

	file << "# vtk DataFile Version 3.0" << std::endl
		<< "Comment if needed" << std::endl;

	file << "BINARY" << std::endl;

	uint64_t pointCount = dxsize * dysize * dzsize * bsize;

	file << "DATASET POLYDATA" << std::endl << "POINTS " << pointCount << " float" << std::endl;

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
					float3 globalPos = float3{ (float)myFloatGlobalPos.x, (float)myFloatGlobalPos.y, (float)myFloatGlobalPos.z };

					swapEnd(globalPos.x);
					swapEnd(globalPos.y);
					swapEnd(globalPos.z);

					file.write((char*)&globalPos.x, sizeof(float));
					file.write((char*)&globalPos.y, sizeof(float));
					file.write((char*)&globalPos.z, sizeof(float));
				}
			}
		}
	}

	file << std::endl << "POINT_DATA " << pointCount << std::endl;
	file << "SCALARS density float 1" << std::endl;
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
					myFloat norm_s1 = pPsi[idx].values[dualNode].s1.x * pPsi[idx].values[dualNode].s1.x + pPsi[idx].values[dualNode].s1.y * pPsi[idx].values[dualNode].s1.y;
					myFloat norm_s0 = pPsi[idx].values[dualNode].s0.x * pPsi[idx].values[dualNode].s0.x + pPsi[idx].values[dualNode].s0.y * pPsi[idx].values[dualNode].s0.y;
					myFloat norm_s_1 = pPsi[idx].values[dualNode].s_1.x * pPsi[idx].values[dualNode].s_1.x + pPsi[idx].values[dualNode].s_1.y * pPsi[idx].values[dualNode].s_1.y;

					float density = (float)(norm_s1 + norm_s0 + norm_s_1);
					swapEnd(density);
					file.write((char*)&density, sizeof(float));
				}
			}
		}
	}

	//file << std::endl << "SCALARS s0 float 1" << std::endl;
	//file << "LOOKUP_TABLE default" << std::endl;
	//
	//for (uint z = 0; z < dzsize; ++z)
	//{
	//	for (uint x = 0; x < dxsize; ++x)
	//	{
	//		for (uint y = 0; y < dysize; ++y)
	//		{
	//			const uint idx = z * dxsize * dysize + y * dxsize + x;
	//			for (uint dualNode = 0; dualNode < VALUES_IN_BLOCK; ++dualNode)
	//			{
	//				myFloat norm_s0 = pPsi[idx].values[dualNode].s0.x * pPsi[idx].values[dualNode].s0.x + pPsi[idx].values[dualNode].s0.y * pPsi[idx].values[dualNode].s0.y;
	//
	//				float density = (float)(norm_s0);
	//				swapEnd(density);
	//				file.write((char*)&density, sizeof(float));
	//			}
	//		}
	//	}
	//}
	//
	//file << std::endl << "SCALARS s-1 float 1" << std::endl;
	//file << "LOOKUP_TABLE default" << std::endl;
	//
	//for (uint z = 0; z < dzsize; ++z)
	//{
	//	for (uint x = 0; x < dxsize; ++x)
	//	{
	//		for (uint y = 0; y < dysize; ++y)
	//		{
	//			const uint idx = z * dxsize * dysize + y * dxsize + x;
	//			for (uint dualNode = 0; dualNode < VALUES_IN_BLOCK; ++dualNode)
	//			{
	//				myFloat norm_s_1 = pPsi[idx].values[dualNode].s_1.x * pPsi[idx].values[dualNode].s_1.x + pPsi[idx].values[dualNode].s_1.y * pPsi[idx].values[dualNode].s_1.y;
	//
	//				float density = (float)(norm_s_1);
	//				swapEnd(density);
	//				file.write((char*)&density, sizeof(float));
	//			}
	//		}
	//	}
	//}

	file << std::endl << "SCALARS spinNorm float 1" << std::endl;
	file << "LOOKUP_TABLE default" << std::endl;

	size_t xStride = dxsize - 2;
	size_t yStride = dysize - 2;

	for (uint z = 0; z < dzsize; ++z)
	{
		for (uint x = 0; x < dxsize; ++x)
		{
			for (uint y = 0; y < dysize; ++y)
			{
				const uint psiIdx = z * dxsize * dysize + y * dxsize + x;
				for (uint dualNode = 0; dualNode < VALUES_IN_BLOCK; ++dualNode)
				{
					myFloat3 avgLocalSpin = { 0, 0, 0 };
					myFloat normSq_s1 = pPsi[psiIdx].values[dualNode].s1.x * pPsi[psiIdx].values[dualNode].s1.x + pPsi[psiIdx].values[dualNode].s1.y * pPsi[psiIdx].values[dualNode].s1.y;
					myFloat normSq_s0 = pPsi[psiIdx].values[dualNode].s0.x * pPsi[psiIdx].values[dualNode].s0.x + pPsi[psiIdx].values[dualNode].s0.y * pPsi[psiIdx].values[dualNode].s0.y;
					myFloat normSq_s_1 = pPsi[psiIdx].values[dualNode].s_1.x * pPsi[psiIdx].values[dualNode].s_1.x + pPsi[psiIdx].values[dualNode].s_1.y * pPsi[psiIdx].values[dualNode].s_1.y;
					myFloat density = normSq_s1 + normSq_s0 + normSq_s_1;

					if ((z > 0) && (y > 0) && (x > 0) &&
						(z < dzsize - 1) && (y < dysize - 1) && (x < dxsize - 1))
					{
						const size_t idx = VALUES_IN_BLOCK * ((z - 1) * xStride * yStride + (y - 1) * xStride + (x - 1)) + dualNode;
						avgLocalSpin = pLocalAvgSpin[idx];
					}

					float spinNorm = 0;
					myFloat3 localPos = getLocalPos(dualNode);
					myFloat3 globalPos = { p0.x + block_scale * (x * BLOCK_WIDTH_X + localPos.x),
						p0.y + block_scale * (y * BLOCK_WIDTH_Y + localPos.y),
						p0.z + block_scale * (z * BLOCK_WIDTH_Z + localPos.z) };
					myFloat distance = sqrt(globalPos.x * globalPos.x + globalPos.y * globalPos.y + globalPos.z * globalPos.z);
					//if (distance < DISTANCE_THRESHOLD)
					if (density > DENSITY_THRESHOLD)
					{
						spinNorm = sqrt(avgLocalSpin.x * avgLocalSpin.x + avgLocalSpin.y * avgLocalSpin.y + avgLocalSpin.z * avgLocalSpin.z);
					}
					swapEnd(spinNorm);

					file.write((char*)&spinNorm, sizeof(float));
				}
			}
		}
	}

	file << std::endl << "SCALARS theta float 1" << std::endl;
	file << "LOOKUP_TABLE default" << std::endl;

	for (uint z = 0; z < dzsize; ++z)
	{
		for (uint x = 0; x < dxsize; ++x)
		{
			for (uint y = 0; y < dysize; ++y)
			{
				const uint psiIdx = z * dxsize * dysize + y * dxsize + x;
				for (uint dualNode = 0; dualNode < VALUES_IN_BLOCK; ++dualNode)
				{
					float theta = 0;
					myFloat normSq_s1 = pPsi[psiIdx].values[dualNode].s1.x * pPsi[psiIdx].values[dualNode].s1.x + pPsi[psiIdx].values[dualNode].s1.y * pPsi[psiIdx].values[dualNode].s1.y;
					myFloat normSq_s0 = pPsi[psiIdx].values[dualNode].s0.x * pPsi[psiIdx].values[dualNode].s0.x + pPsi[psiIdx].values[dualNode].s0.y * pPsi[psiIdx].values[dualNode].s0.y;
					myFloat normSq_s_1 = pPsi[psiIdx].values[dualNode].s_1.x * pPsi[psiIdx].values[dualNode].s_1.x + pPsi[psiIdx].values[dualNode].s_1.y * pPsi[psiIdx].values[dualNode].s_1.y;
					myFloat density = normSq_s1 + normSq_s0 + normSq_s_1;

					if ((density > DENSITY_THRESHOLD) && (z > 0) && (y > 0) && (x > 0) &&
						(z < dzsize - 1) && (y < dysize - 1) && (x < dxsize - 1))
					{
						const size_t idx = VALUES_IN_BLOCK * ((z - 1) * xStride * yStride + (y - 1) * xStride + (x - 1)) + dualNode;
						theta = pTheta[idx];
					}

					swapEnd(theta);

					file.write((char*)&theta, sizeof(float));
				}
			}
		}
	}

	file << std::endl << "VECTORS localAvgSpin float" << std::endl;

	for (uint z = 0; z < dzsize; ++z)
	{
		for (uint x = 0; x < dxsize; ++x)
		{
			for (uint y = 0; y < dysize; ++y)
			{
				const uint psiIdx = z * dxsize * dysize + y * dxsize + x;
				for (uint dualNode = 0; dualNode < VALUES_IN_BLOCK; ++dualNode)
				{
					myFloat normSq_s1 = pPsi[psiIdx].values[dualNode].s1.x * pPsi[psiIdx].values[dualNode].s1.x + pPsi[psiIdx].values[dualNode].s1.y * pPsi[psiIdx].values[dualNode].s1.y;
					myFloat normSq_s0 = pPsi[psiIdx].values[dualNode].s0.x * pPsi[psiIdx].values[dualNode].s0.x + pPsi[psiIdx].values[dualNode].s0.y * pPsi[psiIdx].values[dualNode].s0.y;
					myFloat normSq_s_1 = pPsi[psiIdx].values[dualNode].s_1.x * pPsi[psiIdx].values[dualNode].s_1.x + pPsi[psiIdx].values[dualNode].s_1.y * pPsi[psiIdx].values[dualNode].s_1.y;
					myFloat density = normSq_s1 + normSq_s0 + normSq_s_1;

					float sx = 0;
					float sy = 0;
					float sz = 0;

					if ((density > DENSITY_THRESHOLD) && (z > 0) && (y > 0) && (x > 0) &&
						(z < dzsize - 1) && (y < dysize - 1) && (x < dxsize - 1))
					{
						const size_t idx = VALUES_IN_BLOCK * ((z - 1) * xStride * yStride + (y - 1) * xStride + (x - 1)) + dualNode;
						myFloat3 avgLocalSpin = pLocalAvgSpin[idx];

						sx = avgLocalSpin.x;
						sy = avgLocalSpin.y;
						sz = avgLocalSpin.z;
					}

					swapEnd(sx);
					swapEnd(sy);
					swapEnd(sz);

					file.write((char*)&sx, sizeof(float));
					file.write((char*)&sy, sizeof(float));
					file.write((char*)&sz, sizeof(float));
				}
			}
		}
	}

	file << std::endl << "VECTORS u float" << std::endl;

	for (uint z = 0; z < dzsize; ++z)
	{
		for (uint x = 0; x < dxsize; ++x)
		{
			for (uint y = 0; y < dysize; ++y)
			{
				const uint psiIdx = z * dxsize * dysize + y * dxsize + x;
				for (uint dualNode = 0; dualNode < VALUES_IN_BLOCK; ++dualNode)
				{
					myFloat normSq_s1 = pPsi[psiIdx].values[dualNode].s1.x * pPsi[psiIdx].values[dualNode].s1.x + pPsi[psiIdx].values[dualNode].s1.y * pPsi[psiIdx].values[dualNode].s1.y;
					myFloat normSq_s0 = pPsi[psiIdx].values[dualNode].s0.x * pPsi[psiIdx].values[dualNode].s0.x + pPsi[psiIdx].values[dualNode].s0.y * pPsi[psiIdx].values[dualNode].s0.y;
					myFloat normSq_s_1 = pPsi[psiIdx].values[dualNode].s_1.x * pPsi[psiIdx].values[dualNode].s_1.x + pPsi[psiIdx].values[dualNode].s_1.y * pPsi[psiIdx].values[dualNode].s_1.y;
					myFloat density = normSq_s1 + normSq_s0 + normSq_s_1;

					float ux = 0;
					float uy = 0;
					float uz = 0;

					if ((density > DENSITY_THRESHOLD) && (z > 0) && (y > 0) && (x > 0) &&
						(z < dzsize - 1) && (y < dysize - 1) && (x < dxsize - 1))
					{
						const size_t idx = VALUES_IN_BLOCK * ((z - 1) * xStride * yStride + (y - 1) * xStride + (x - 1)) + dualNode;
						myFloat3 u = pu[idx];

						ux = u.x;
						uy = u.y;
						uz = u.z;
					}

					swapEnd(ux);
					swapEnd(uy);
					swapEnd(uz);

					file.write((char*)&ux, sizeof(float));
					file.write((char*)&uy, sizeof(float));
					file.write((char*)&uz, sizeof(float));
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

void savePreImageSpinor(const std::string& folder, BlockPsis* pPsi, size_t bsize, size_t dxsize, size_t dysize, size_t dzsize, myFloat block_scale, myFloat3 p0, myFloat t)
{
	constexpr myFloat EPSILON = 0.0005;

	std::ofstream file;
	file.open(folder + "/pre_image_" + std::to_string(t) + ".vtk", std::ios::out | std::ios::binary);

	file << "# vtk DataFile Version 3.0" << std::endl
		<< "Comment if needed" << std::endl;

	file << "BINARY" << std::endl;

	uint64_t pointCount = 0;
	uint64_t otherPointCount = 0;
	for (uint z = 0; z < dzsize; ++z)
	{
		for (uint x = 0; x < dxsize; ++x)
		{
			for (uint y = 0; y < dysize; ++y)
			{
				const uint idx = z * dxsize * dysize + y * dxsize + x;
				for (uint dualNode = 0; dualNode < VALUES_IN_BLOCK; ++dualNode)
				{
					if ((z > 0) && (y > 0) && (x > 0) &&
						(z < dzsize - 1) && (y < dysize - 1) && (x < dxsize - 1))
					{
						myFloat2 s1 = pPsi[idx].values[dualNode].s1;
						myFloat2 s0 = pPsi[idx].values[dualNode].s0;
						myFloat2 s_1 = pPsi[idx].values[dualNode].s_1;

						myFloat normSq_s1 = s1.x * s1.x + s1.y * s1.y;
						myFloat normSq_s_1 = s_1.x * s_1.x + s_1.y * s_1.y;

						myFloat2 temp = sqrt(2) * (conj(s1) * s0 + conj(s0) * s_1);
						myFloat3 avgLocalSpin = { temp.x, temp.y, normSq_s1 - normSq_s_1 };
						avgLocalSpin = avgLocalSpin / sqrt(avgLocalSpin.x * avgLocalSpin.x + avgLocalSpin.y * avgLocalSpin.y + avgLocalSpin.z * avgLocalSpin.z);
						if (abs(1.0 - abs(avgLocalSpin.y)) < EPSILON) // Take the spinors whose normalized local spin is almost only +/- y-direction (i.e. y ~= 1)
						{
							pointCount++;
						}
						else
						{
							otherPointCount++;
						}
					}
				}
			}
		}
	}
	//std::cout << "Point count is " << pointCount << std::endl;
	//std::cout << "Other count is " << otherPointCount << std::endl;

	file << "DATASET POLYDATA" << std::endl << "POINTS " << pointCount << " float" << std::endl;

	for (uint z = 0; z < dzsize; ++z)
	{
		for (uint x = 0; x < dxsize; ++x)
		{
			for (uint y = 0; y < dysize; ++y)
			{
				const uint idx = z * dxsize * dysize + y * dxsize + x;
				for (uint dualNode = 0; dualNode < VALUES_IN_BLOCK; ++dualNode)
				{
					if ((z > 0) && (y > 0) && (x > 0) &&
						(z < dzsize - 1) && (y < dysize - 1) && (x < dxsize - 1))
					{
						myFloat2 s1 = pPsi[idx].values[dualNode].s1;
						myFloat2 s0 = pPsi[idx].values[dualNode].s0;
						myFloat2 s_1 = pPsi[idx].values[dualNode].s_1;

						myFloat normSq_s1 = s1.x * s1.x + s1.y * s1.y;
						myFloat normSq_s_1 = s_1.x * s_1.x + s_1.y * s_1.y;

						myFloat2 temp = sqrt(2) * (conj(s1) * s0 + conj(s0) * s_1);
						myFloat3 avgLocalSpin = { temp.x, temp.y, normSq_s1 - normSq_s_1 };
						avgLocalSpin = avgLocalSpin / sqrt(avgLocalSpin.x * avgLocalSpin.x + avgLocalSpin.y * avgLocalSpin.y + avgLocalSpin.z * avgLocalSpin.z);
						if (abs(1.0 - abs(avgLocalSpin.y)) < EPSILON) // Take the spinors whose normalized local spin is almost only +/- y-direction (i.e. y ~= 1)
						{
							myFloat3 localPos = getLocalPos(dualNode);
							myFloat3 myFloatGlobalPos = { p0.x + block_scale * (x * BLOCK_WIDTH_X + localPos.x),
								p0.y + block_scale * (y * BLOCK_WIDTH_Y + localPos.y),
								p0.z + block_scale * (z * BLOCK_WIDTH_Z + localPos.z) };
							float3 globalPos = float3{ (float)myFloatGlobalPos.x, (float)myFloatGlobalPos.y, (float)myFloatGlobalPos.z };

							swapEnd(globalPos.x);
							swapEnd(globalPos.y);
							swapEnd(globalPos.z);

							file.write((char*)&globalPos.x, sizeof(float));
							file.write((char*)&globalPos.y, sizeof(float));
							file.write((char*)&globalPos.z, sizeof(float));
						}
					}
				}
			}
		}
	}

	file << std::endl << "POINT_DATA " << pointCount << std::endl;

	file << std::endl << "SCALARS r_m=1 float 1" << std::endl;
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
					if ((z > 0) && (y > 0) && (x > 0) &&
						(z < dzsize - 1) && (y < dysize - 1) && (x < dxsize - 1))
					{
						myFloat2 s1 = pPsi[idx].values[dualNode].s1;
						myFloat2 s0 = pPsi[idx].values[dualNode].s0;
						myFloat2 s_1 = pPsi[idx].values[dualNode].s_1;

						myFloat normSq_s1 = s1.x * s1.x + s1.y * s1.y;
						myFloat normSq_s_1 = s_1.x * s_1.x + s_1.y * s_1.y;

						myFloat2 temp = sqrt(2) * (conj(s1) * s0 + conj(s0) * s_1);
						myFloat3 avgLocalSpin = { temp.x, temp.y, normSq_s1 - normSq_s_1 };
						avgLocalSpin = avgLocalSpin / sqrt(avgLocalSpin.x * avgLocalSpin.x + avgLocalSpin.y * avgLocalSpin.y + avgLocalSpin.z * avgLocalSpin.z);
						if (abs(1.0 - abs(avgLocalSpin.y)) < EPSILON) // Take the spinors whose normalized local spin is almost only +/- y-direction (i.e. y ~= 1)
						{
							myFloat dens_s1 = s1.x * s1.x + s1.y * s1.y;
							myFloat dens_s0 = s0.x * s0.x + s0.y * s0.y;
							myFloat dens_s_1 = s_1.x * s_1.x + s_1.y * s_1.y;
							myFloat dens = dens_s1 + dens_s0 + dens_s_1;

							float s1_r = 0;
							if (DENSITY_THRESHOLD < dens)
								s1_r = (float)(s1.x / sqrt(dens));
							swapEnd(s1_r);
							file.write((char*)&s1_r, sizeof(float));
						}
					}
				}
			}
		}
	}
	file << std::endl << "SCALARS i_m=1 float 1" << std::endl;
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
					if ((z > 0) && (y > 0) && (x > 0) &&
						(z < dzsize - 1) && (y < dysize - 1) && (x < dxsize - 1))
					{
						myFloat2 s1 = pPsi[idx].values[dualNode].s1;
						myFloat2 s0 = pPsi[idx].values[dualNode].s0;
						myFloat2 s_1 = pPsi[idx].values[dualNode].s_1;

						myFloat normSq_s1 = s1.x * s1.x + s1.y * s1.y;
						myFloat normSq_s_1 = s_1.x * s_1.x + s_1.y * s_1.y;

						myFloat2 temp = sqrt(2) * (conj(s1) * s0 + conj(s0) * s_1);
						myFloat3 avgLocalSpin = { temp.x, temp.y, normSq_s1 - normSq_s_1 };
						avgLocalSpin = avgLocalSpin / sqrt(avgLocalSpin.x * avgLocalSpin.x + avgLocalSpin.y * avgLocalSpin.y + avgLocalSpin.z * avgLocalSpin.z);
						if (abs(1.0 - abs(avgLocalSpin.y)) < EPSILON) // Take the spinors whose normalized local spin is almost only +/- y-direction (i.e. y ~= 1)
						{
							myFloat dens_s1 = s1.x * s1.x + s1.y * s1.y;
							myFloat dens_s0 = s0.x * s0.x + s0.y * s0.y;
							myFloat dens_s_1 = s_1.x * s_1.x + s_1.y * s_1.y;
							myFloat dens = dens_s1 + dens_s0 + dens_s_1;

							float s1_i = 0;
							if (DENSITY_THRESHOLD < dens)
								s1_i = (float)(s1.y / sqrt(dens));
							swapEnd(s1_i);
							file.write((char*)&s1_i, sizeof(float));
						}
					}
				}
			}
		}
	}

	file << std::endl << "SCALARS r_m=0 float 1" << std::endl;
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
					if ((z > 0) && (y > 0) && (x > 0) &&
						(z < dzsize - 1) && (y < dysize - 1) && (x < dxsize - 1))
					{
						myFloat2 s1 = pPsi[idx].values[dualNode].s1;
						myFloat2 s0 = pPsi[idx].values[dualNode].s0;
						myFloat2 s_1 = pPsi[idx].values[dualNode].s_1;

						myFloat normSq_s1 = s1.x * s1.x + s1.y * s1.y;
						myFloat normSq_s_1 = s_1.x * s_1.x + s_1.y * s_1.y;

						myFloat2 temp = sqrt(2) * (conj(s1) * s0 + conj(s0) * s_1);
						myFloat3 avgLocalSpin = { temp.x, temp.y, normSq_s1 - normSq_s_1 };
						avgLocalSpin = avgLocalSpin / sqrt(avgLocalSpin.x * avgLocalSpin.x + avgLocalSpin.y * avgLocalSpin.y + avgLocalSpin.z * avgLocalSpin.z);
						if (abs(1.0 - abs(avgLocalSpin.y)) < EPSILON) // Take the spinors whose normalized local spin is almost only +/- y-direction (i.e. y ~= 1)
						{
							myFloat dens_s1 = s1.x * s1.x + s1.y * s1.y;
							myFloat dens_s0 = s0.x * s0.x + s0.y * s0.y;
							myFloat dens_s_1 = s_1.x * s_1.x + s_1.y * s_1.y;
							myFloat dens = dens_s1 + dens_s0 + dens_s_1;

							float s0_r = 0;
							if (DENSITY_THRESHOLD < dens)
								s0_r = (float)(s0.x / sqrt(dens));
							swapEnd(s0_r);
							file.write((char*)&s0_r, sizeof(float));
						}
					}
				}
			}
		}
	}
	file << std::endl << "SCALARS i_m=0 float 1" << std::endl;
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
					if ((z > 0) && (y > 0) && (x > 0) &&
						(z < dzsize - 1) && (y < dysize - 1) && (x < dxsize - 1))
					{
						myFloat2 s1 = pPsi[idx].values[dualNode].s1;
						myFloat2 s0 = pPsi[idx].values[dualNode].s0;
						myFloat2 s_1 = pPsi[idx].values[dualNode].s_1;

						myFloat normSq_s1 = s1.x * s1.x + s1.y * s1.y;
						myFloat normSq_s_1 = s_1.x * s_1.x + s_1.y * s_1.y;

						myFloat2 temp = sqrt(2) * (conj(s1) * s0 + conj(s0) * s_1);
						myFloat3 avgLocalSpin = { temp.x, temp.y, normSq_s1 - normSq_s_1 };
						avgLocalSpin = avgLocalSpin / sqrt(avgLocalSpin.x * avgLocalSpin.x + avgLocalSpin.y * avgLocalSpin.y + avgLocalSpin.z * avgLocalSpin.z);
						if (abs(1.0 - abs(avgLocalSpin.y)) < EPSILON) // Take the spinors whose normalized local spin is almost only +/- y-direction (i.e. y ~= 1)
						{
							myFloat dens_s1 = s1.x * s1.x + s1.y * s1.y;
							myFloat dens_s0 = s0.x * s0.x + s0.y * s0.y;
							myFloat dens_s_1 = s_1.x * s_1.x + s_1.y * s_1.y;
							myFloat dens = dens_s1 + dens_s0 + dens_s_1;

							float s0_i = 0;
							if (DENSITY_THRESHOLD < dens)
								s0_i = (float)(s0.y / sqrt(dens));
							swapEnd(s0_i);
							file.write((char*)&s0_i, sizeof(float));
						}
					}
				}
			}
		}
	}

	file << std::endl << "SCALARS r_m=-1 float 1" << std::endl;
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
					if ((z > 0) && (y > 0) && (x > 0) &&
						(z < dzsize - 1) && (y < dysize - 1) && (x < dxsize - 1))
					{
						myFloat2 s1 = pPsi[idx].values[dualNode].s1;
						myFloat2 s0 = pPsi[idx].values[dualNode].s0;
						myFloat2 s_1 = pPsi[idx].values[dualNode].s_1;

						myFloat normSq_s1 = s1.x * s1.x + s1.y * s1.y;
						myFloat normSq_s_1 = s_1.x * s_1.x + s_1.y * s_1.y;

						myFloat2 temp = sqrt(2) * (conj(s1) * s0 + conj(s0) * s_1);
						myFloat3 avgLocalSpin = { temp.x, temp.y, normSq_s1 - normSq_s_1 };
						avgLocalSpin = avgLocalSpin / sqrt(avgLocalSpin.x * avgLocalSpin.x + avgLocalSpin.y * avgLocalSpin.y + avgLocalSpin.z * avgLocalSpin.z);
						if (abs(1.0 - abs(avgLocalSpin.y)) < EPSILON) // Take the spinors whose normalized local spin is almost only +/- y-direction (i.e. y ~= 1)
						{
							myFloat dens_s1 = s1.x * s1.x + s1.y * s1.y;
							myFloat dens_s0 = s0.x * s0.x + s0.y * s0.y;
							myFloat dens_s_1 = s_1.x * s_1.x + s_1.y * s_1.y;
							myFloat dens = dens_s1 + dens_s0 + dens_s_1;

							float s_1_r = 0;
							if (DENSITY_THRESHOLD < dens)
								s_1_r = (float)(s_1.x / sqrt(dens));
							swapEnd(s_1_r);
							file.write((char*)&s_1_r, sizeof(float));
						}
					}
				}
			}
		}
	}
	file << std::endl << "SCALARS i_m=-1 float 1" << std::endl;
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
					if ((z > 0) && (y > 0) && (x > 0) &&
						(z < dzsize - 1) && (y < dysize - 1) && (x < dxsize - 1))
					{
						myFloat2 s1 = pPsi[idx].values[dualNode].s1;
						myFloat2 s0 = pPsi[idx].values[dualNode].s0;
						myFloat2 s_1 = pPsi[idx].values[dualNode].s_1;

						myFloat normSq_s1 = s1.x * s1.x + s1.y * s1.y;
						myFloat normSq_s_1 = s_1.x * s_1.x + s_1.y * s_1.y;

						myFloat2 temp = sqrt(2) * (conj(s1) * s0 + conj(s0) * s_1);
						myFloat3 avgLocalSpin = { temp.x, temp.y, normSq_s1 - normSq_s_1 };
						avgLocalSpin = avgLocalSpin / sqrt(avgLocalSpin.x * avgLocalSpin.x + avgLocalSpin.y * avgLocalSpin.y + avgLocalSpin.z * avgLocalSpin.z);
						if (abs(1.0 - abs(avgLocalSpin.y)) < EPSILON) // Take the spinors whose normalized local spin is almost only +/- y-direction (i.e. y ~= 1)
						{
							myFloat dens_s1 = s1.x * s1.x + s1.y * s1.y;
							myFloat dens_s0 = s0.x * s0.x + s0.y * s0.y;
							myFloat dens_s_1 = s_1.x * s_1.x + s_1.y * s_1.y;
							myFloat dens = dens_s1 + dens_s0 + dens_s_1;

							float s_1_i = 0;
							if (DENSITY_THRESHOLD < dens)
								s_1_i = (float)(s_1.y / sqrt(dens));
							swapEnd(s_1_i);
							file.write((char*)&s_1_i, sizeof(float));
						}
					}
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

#endif // UTILS