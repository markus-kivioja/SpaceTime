#ifndef UTILS
#define UTILS

#include <cuda_runtime.h>
#include "helper_cuda.h"

#include "Output/Picture.hpp"
#include "Output/Text.hpp"
#include "Types/Complex.hpp"

#include "mesh.h"

// Arithmetic operators for cuda vector types
__host__ __device__ __inline__ double2 operator+(double2 a, double2 b)
{
	return { a.x + b.x, a.y + b.y };
}
__host__ __device__ __inline__ double3 operator+(double3 a, double3 b)
{
	return { a.x + b.x, a.y + b.y, a.z + b.z };
}
__host__ __device__ __inline__ double2 operator-(double2 a, double2 b)
{
	return { a.x - b.x, a.y - b.y };
}
__host__ __device__ __inline__ double2 operator-(double2 a)
{
	return { -a.x, -a.y };
}
__host__ __device__ __inline__ double3 operator-(double3 a, double3 b)
{
	return { a.x - b.x, a.y - b.y, a.z - b.z };
}
__host__ __device__ __inline__ void operator+=(double2& a, double2 b)
{
	a.x += b.x;
	a.y += b.y;
}
__host__ __device__ __inline__ void operator+=(double3& a, double3 b)
{
	a.x += b.x;
	a.y += b.y;
	a.z += b.z;
}
__host__ __device__ __inline__ void operator-=(double2& a, double2 b)
{
	a.x -= b.x;
	a.y -= b.y;
}
__host__ __device__ __inline__ double2 operator*(double b, double2 a)
{
	return { b * a.x, b * a.y };
}
__host__ __device__ __inline__ double3 operator*(double b, double3 a)
{
	return { b * a.x, b * a.y, b * a.z };
}
__host__ __device__ __inline__ double3 operator*(double3 a, double b)
{
	return { b * a.x, b * a.y, b * a.z };
}
__host__ __device__ __inline__ double3 operator/(double3 a, double b)
{
	return { a.x / b, a.y / b, a.z / b };
}
__host__ __device__ __inline__ double2 operator/(double2 a, double b)
{
	return { a.x / b, a.y / b };
}
__host__ __device__ __inline__ double2 conj(double2 a) // Complex conjugate
{
	return { a.x, -a.y };
}
__host__ __device__ __inline__ double2 operator*(double2 a, double2 b) // Complex number multiplication
{
	return { a.x * b.x - a.y * b.y, a.y * b.x + a.x * b.y };
}

struct Complex3Vec
{
	double2 s1;
	double2 s0;
	double2 s_1;
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
	double Bq{};
	double3 Bb{};
	double BqQuad{};
	double3 BbQuad{};
};

std::string toString(const double value)
{
	std::ostringstream out;
	out.precision(18);
	out << std::fixed << value;
	return out.str();
};

void drawDensity(const std::string& filePrefix, BlockPsis* h_evenPsi, size_t dxsize, size_t dysize, size_t dzsize, double t, const std::string& folder)
{
	const int SIZE = 2;
	double INTENSITY = 1.0;

	const int width = dxsize * SIZE, height = dysize * SIZE, depth = dzsize * SIZE;
	Picture pic1(width * 3, height * 2);

	double maxVal = 0;
	// XZ-plane
	for (uint k = 0; k < depth; ++k)
	{
		for (uint i = 0; i < width; i++)
		{
			double norm_s1 = 0;
			double norm_s0 = 0;
			double norm_s_1 = 0;
			for (uint j = 0; j < height; j++)
			{
				const uint idx = (k / SIZE) * dxsize * dysize + (j / SIZE) * dxsize + i / SIZE;
				for (uint dualNode = 0; dualNode < VALUES_IN_BLOCK; ++dualNode)
				{
					double2 s1 = h_evenPsi[idx].values[dualNode].s1;
					double2 s0 = h_evenPsi[idx].values[dualNode].s0;
					double2 s_1 = h_evenPsi[idx].values[dualNode].s_1;

#if BASIS == X_QUANTIZED
					double2 x_s1 = 0.5 * (s1 + s_1) - s0 / sqrt(2);
					double2 x_s0 = (s1 - s_1) / sqrt(2);
					double2 x_s_1 = 0.5 * (s_1 + s1) + s0 / sqrt(2);

					s1 = x_s1;
					s0 = x_s0;
					s_1 = x_s_1;
#elif BASIS == Y_QUANTIZED
					double2 y_s1 = 0.5 * (s1 - s_1) + s0 * double2{ 0, -1 } / sqrt(2);
					double2 y_s0 = (s1 + s_1) * double2{ 0, -1 } / sqrt(2);
					double2 y_s_1 = 0.5 * (s_1 - s1) + s0 * double2{ 0, -1 } / sqrt(2);

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
			double norm_s1 = 0;
			double norm_s0 = 0;
			double norm_s_1 = 0;
			for (uint j = 0; j < height; j++)
			{
				const uint idx = (k / SIZE) * dxsize * dysize + (j / SIZE) * dxsize + i / SIZE;
				for (uint dualNode = 0; dualNode < VALUES_IN_BLOCK; ++dualNode)
				{
					double2 s1 = h_evenPsi[idx].values[dualNode].s1;
					double2 s0 = h_evenPsi[idx].values[dualNode].s0;
					double2 s_1 = h_evenPsi[idx].values[dualNode].s_1;

#if BASIS == X_QUANTIZED
					double2 x_s1 = 0.5 * (s1 + s_1) - s0 / sqrt(2);
					double2 x_s0 = (s1 - s_1) / sqrt(2);
					double2 x_s_1 = 0.5 * (s_1 + s1) + s0 / sqrt(2);

					s1 = x_s1;
					s0 = x_s0;
					s_1 = x_s_1;
#elif BASIS == Y_QUANTIZED
					double2 y_s1 = 0.5 * (s1 - s_1) + s0 * double2{ 0, -1 } / sqrt(2);
					double2 y_s0 = (s1 + s_1) * double2{ 0, -1 } / sqrt(2);
					double2 y_s_1 = 0.5 * (s_1 - s1) + s0 * double2{ 0, -1 } / sqrt(2);

					s1 = y_s1;
					s0 = y_s0;
					s_1 = y_s_1;
#endif

					norm_s1 += s1.x * s1.x + s1.y * s1.y;
					norm_s0 += s0.x * s0.x + s0.y * s0.y;
					norm_s_1 += s_1.x * s_1.x + s_1.y * s_1.y;
				}
			}

			const double s1 = INTENSITY * norm_s1;
			const double s0 = INTENSITY * norm_s0;
			const double s_1 = INTENSITY * norm_s_1;

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
			double norm_s1 = 0;
			double norm_s0 = 0;
			double norm_s_1 = 0;
			for (uint k = 0; k < depth; ++k)
			{
				const uint idx = (k / SIZE) * dxsize * dysize + (j / SIZE) * dxsize + i / SIZE;
				for (uint dualNode = 0; dualNode < VALUES_IN_BLOCK; ++dualNode)
				{
					double2 s1 = h_evenPsi[idx].values[dualNode].s1;
					double2 s0 = h_evenPsi[idx].values[dualNode].s0;
					double2 s_1 = h_evenPsi[idx].values[dualNode].s_1;

#if BASIS == X_QUANTIZED
					double2 x_s1 = 0.5 * (s1 + s_1) - s0 / sqrt(2);
					double2 x_s0 = (s1 - s_1) / sqrt(2);
					double2 x_s_1 = 0.5 * (s_1 + s1) + s0 / sqrt(2);

					s1 = x_s1;
					s0 = x_s0;
					s_1 = x_s_1;
#elif BASIS == Y_QUANTIZED
					double2 y_s1 = 0.5 * (s1 - s_1) + s0 * double2{ 0, -1 } / sqrt(2);
					double2 y_s0 = (s1 + s_1) * double2{ 0, -1 } / sqrt(2);
					double2 y_s_1 = 0.5 * (s_1 - s1) + s0 * double2{ 0, -1 } / sqrt(2);

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
			double norm_s1 = 0;
			double norm_s0 = 0;
			double norm_s_1 = 0;
			for (uint k = 0; k < depth; ++k)
			{
				const uint idx = (k / SIZE) * dxsize * dysize + (j / SIZE) * dxsize + i / SIZE;
				for (uint dualNode = 0; dualNode < VALUES_IN_BLOCK; ++dualNode)
				{
					double2 s1 = h_evenPsi[idx].values[dualNode].s1;
					double2 s0 = h_evenPsi[idx].values[dualNode].s0;
					double2 s_1 = h_evenPsi[idx].values[dualNode].s_1;

#if BASIS == X_QUANTIZED
					double2 x_s1 = 0.5 * (s1 + s_1) - s0 / sqrt(2);
					double2 x_s0 = (s1 - s_1) / sqrt(2);
					double2 x_s_1 = 0.5 * (s_1 + s1) + s0 / sqrt(2);

					s1 = x_s1;
					s0 = x_s0;
					s_1 = x_s_1;
#elif BASIS == Y_QUANTIZED
					double2 y_s1 = 0.5 * (s1 - s_1) + s0 * double2{ 0, -1 } / sqrt(2);
					double2 y_s0 = (s1 + s_1) * double2{ 0, -1 } / sqrt(2);
					double2 y_s_1 = 0.5 * (s_1 - s1) + s0 * double2{ 0, -1 } / sqrt(2);

					s1 = y_s1;
					s0 = y_s0;
					s_1 = y_s_1;
#endif

					norm_s1 += s1.x * s1.x + s1.y * s1.y;
					norm_s0 += s0.x * s0.x + s0.y * s0.y;
					norm_s_1 += s_1.x * s_1.x + s_1.y * s_1.y;
				}
			}

			const double s1 = INTENSITY * norm_s1;
			const double s0 = INTENSITY * norm_s0;
			const double s_1 = INTENSITY * norm_s_1;

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

void drawDensityRI(BlockPsis* h_evenPsi, size_t dxsize, size_t dysize, size_t dzsize, double t, const std::string& folder)
{
	const int SIZE = 2;
	double INTENSITY = 1.0;

	const int width = dxsize * SIZE, height = dysize * SIZE, depth = dzsize * SIZE;
	Picture pic1(width * 3, height * 2);

	double maxVal = 0;
	// XZ-plane
	for (uint k = 0; k < depth; ++k)
	{
		for (uint i = 0; i < width; i++)
		{
			double norm_s1 = 0;
			double norm_s0 = 0;
			double norm_s_1 = 0;
			for (uint j = 0; j < height; j++)
			{
				const uint idx = (k / SIZE) * dxsize * dysize + (j / SIZE) * dxsize + i / SIZE;
				for (uint dualNode = 0; dualNode < VALUES_IN_BLOCK; ++dualNode)
				{
					double2 s1 = h_evenPsi[idx].values[dualNode].s1;
					double2 s0 = h_evenPsi[idx].values[dualNode].s0;
					double2 s_1 = h_evenPsi[idx].values[dualNode].s_1;

#if BASIS == X_QUANTIZED
					double2 x_s1 = 0.5 * (s1 + s_1) - s0 / sqrt(2);
					double2 x_s0 = (s1 - s_1) / sqrt(2);
					double2 x_s_1 = 0.5 * (s_1 + s1) + s0 / sqrt(2);

					s1 = x_s1;
					s0 = x_s0;
					s_1 = x_s_1;
#elif BASIS == Y_QUANTIZED
					double2 y_s1 = 0.5 * (s1 - s_1) + s0 * double2{ 0, -1 } / sqrt(2);
					double2 y_s0 = (s1 + s_1) * double2{ 0, -1 } / sqrt(2);
					double2 y_s_1 = 0.5 * (s_1 - s1) + s0 * double2{ 0, -1 } / sqrt(2);

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
			double2 norm_s1 = double2{ 0, 0 };
			double2 norm_s0 = double2{ 0, 0 };
			double2 norm_s_1 = double2{ 0, 0 };
			for (uint j = 0; j < height; j++)
			{
				const uint idx = (k / SIZE) * dxsize * dysize + (j / SIZE) * dxsize + i / SIZE;
				for (uint dualNode = 0; dualNode < VALUES_IN_BLOCK; ++dualNode)
				{
					double2 s1 = h_evenPsi[idx].values[dualNode].s1;
					double2 s0 = h_evenPsi[idx].values[dualNode].s0;
					double2 s_1 = h_evenPsi[idx].values[dualNode].s_1;

#if BASIS == X_QUANTIZED
					double2 x_s1 = 0.5 * (s1 + s_1) - s0 / sqrt(2);
					double2 x_s0 = (s1 - s_1) / sqrt(2);
					double2 x_s_1 = 0.5 * (s_1 + s1) + s0 / sqrt(2);

					s1 = x_s1;
					s0 = x_s0;
					s_1 = x_s_1;
#elif BASIS == Y_QUANTIZED
					double2 y_s1 = 0.5 * (s1 - s_1) + s0 * double2{ 0, -1 } / sqrt(2);
					double2 y_s0 = (s1 + s_1) * double2{ 0, -1 } / sqrt(2);
					double2 y_s_1 = 0.5 * (s_1 - s1) + s0 * double2{ 0, -1 } / sqrt(2);

					s1 = y_s1;
					s0 = y_s0;
					s_1 = y_s_1;
#endif

					norm_s1 += double2{ s1.x * s1.x, s1.y * s1.y };
					norm_s0 += double2{ s0.x * s0.x, s0.y * s0.y };
					norm_s_1 += double2{ s_1.x * s_1.x, s_1.y * s_1.y };
				}
			}

			const double2 s1 = INTENSITY * norm_s1;
			const double2 s0 = INTENSITY * norm_s0;
			const double2 s_1 = INTENSITY * norm_s_1;

			pic1.setColor(i, k, Vector4(s1.x, s1.y, 0.0, 1.0));
			pic1.setColor(width + i, k, Vector4(s0.x, s0.y, 0.0, 1.0));
			pic1.setColor(2 * width + i, k, Vector4(s_1.x, s_1.y, 0.0, 1.0));
		}
	}

	// XY-plane
	maxVal = 0;
	for (uint j = 0; j < height; j++)
	{
		for (uint i = 0; i < width; i++)
		{
			double norm_s1 = 0;
			double norm_s0 = 0;
			double norm_s_1 = 0;
			for (uint k = 0; k < depth; ++k)
			{
				const uint idx = (k / SIZE) * dxsize * dysize + (j / SIZE) * dxsize + i / SIZE;
				for (uint dualNode = 0; dualNode < VALUES_IN_BLOCK; ++dualNode)
				{
					double2 s1 = h_evenPsi[idx].values[dualNode].s1;
					double2 s0 = h_evenPsi[idx].values[dualNode].s0;
					double2 s_1 = h_evenPsi[idx].values[dualNode].s_1;

#if BASIS == X_QUANTIZED
					double2 x_s1 = 0.5 * (s1 + s_1) - s0 / sqrt(2);
					double2 x_s0 = (s1 - s_1) / sqrt(2);
					double2 x_s_1 = 0.5 * (s_1 + s1) + s0 / sqrt(2);

					s1 = x_s1;
					s0 = x_s0;
					s_1 = x_s_1;
#elif BASIS == Y_QUANTIZED
					double2 y_s1 = 0.5 * (s1 - s_1) + s0 * double2{ 0, -1 } / sqrt(2);
					double2 y_s0 = (s1 + s_1) * double2{ 0, -1 } / sqrt(2);
					double2 y_s_1 = 0.5 * (s_1 - s1) + s0 * double2{ 0, -1 } / sqrt(2);

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
			double2 norm_s1 = double2{ 0, 0 };
			double2 norm_s0 = double2{ 0, 0 };
			double2 norm_s_1 = double2{ 0, 0 };
			for (uint k = 0; k < depth; ++k)
			{
				const uint idx = (k / SIZE) * dxsize * dysize + (j / SIZE) * dxsize + i / SIZE;
				for (uint dualNode = 0; dualNode < VALUES_IN_BLOCK; ++dualNode)
				{
					double2 s1 = h_evenPsi[idx].values[dualNode].s1;
					double2 s0 = h_evenPsi[idx].values[dualNode].s0;
					double2 s_1 = h_evenPsi[idx].values[dualNode].s_1;

#if BASIS == X_QUANTIZED
					double2 x_s1 = 0.5 * (s1 + s_1) - s0 / sqrt(2);
					double2 x_s0 = (s1 - s_1) / sqrt(2);
					double2 x_s_1 = 0.5 * (s_1 + s1) + s0 / sqrt(2);

					s1 = x_s1;
					s0 = x_s0;
					s_1 = x_s_1;
#elif BASIS == Y_QUANTIZED
					double2 y_s1 = 0.5 * (s1 - s_1) + s0 * double2{ 0, -1 } / sqrt(2);
					double2 y_s0 = (s1 + s_1) * double2{ 0, -1 } / sqrt(2);
					double2 y_s_1 = 0.5 * (s_1 - s1) + s0 * double2{ 0, -1 } / sqrt(2);

					s1 = y_s1;
					s0 = y_s0;
					s_1 = y_s_1;
#endif

					norm_s1 += double2{ s1.x * s1.x, s1.y * s1.y };
					norm_s0 += double2{ s0.x * s0.x, s0.y * s0.y };
					norm_s_1 += double2{ s_1.x * s_1.x, s_1.y * s_1.y };
				}
			}

			const double2 s1 = INTENSITY * norm_s1;
			const double2 s0 = INTENSITY * norm_s0;
			const double2 s_1 = INTENSITY * norm_s_1;

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

	pic1.save(folder + "/dens_profiles/" + toString(t) + "ms.bmp", false);
}

void drawDensityRgb(const std::string& name, BlockPsis* h_evenPsi, size_t dxsize, size_t dysize, size_t dzsize, double t, const std::string& folder)
{
	const int SIZE = 4;
	const double INTENSITY = 1.0;

	const int width = dxsize * SIZE, height = dysize * SIZE, depth = dzsize * SIZE;
	Picture pic1(width * 2, height);

	// XZ-plane
	for (uint k = 0; k < depth; ++k)
	{
		for (uint i = 0; i < width; i++)
		{
			double norm_s1 = 0;
			double norm_s0 = 0;
			double norm_s_1 = 0;
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

			const double s1 = INTENSITY * norm_s1;
			const double s0 = INTENSITY * norm_s0;
			const double s_1 = INTENSITY * norm_s_1;

			pic1.setColor(i, k, Vector4(s1, s_1, s0, 1.0));
		}
	}

	// XY-plane
	for (uint j = 0; j < height; j++)
	{
		for (uint i = 0; i < width; i++)
		{
			double norm_s1 = 0;
			double norm_s0 = 0;
			double norm_s_1 = 0;
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

			const double s1 = INTENSITY * norm_s1;
			const double s0 = INTENSITY * norm_s0;
			const double s_1 = INTENSITY * norm_s_1;

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

void drawUtheta(const double3* uPtr, const double* thetaPtr, const size_t xSize, const size_t ySize, const size_t zSize, const double t)
{
	const uint SIZE = 2;
	const double U_INTENSITY = 15.0;
	const double THETA_INTENSITY = 24.0;

	const int width = xSize * SIZE, height = ySize * SIZE, depth = zSize * SIZE;
	Picture pic1(width * 2, height * 2);

	// XZ-plane
	uint y = height / 2;
	for (uint z = 0; z < depth; z += SIZE)
	{
		for (uint x = 0; x < width; x += SIZE)
		{
			double3 us[4] = {
				double3{ 0, 0, 0 },
				double3{ 0, 0, 0 },
				double3{ 0, 0, 0 },
				double3{ 0, 0, 0 }
			};
			double thetas[4] = { 0, 0, 0, 0 };
			uint counts[4] = { 0, 0, 0, 0 };

			for (uint cellIdx = 0; cellIdx < VALUES_IN_BLOCK; ++cellIdx)
			{
				const uint structIdx = VALUES_IN_BLOCK * ((z / SIZE) * xSize * ySize + (y / SIZE) * xSize + (x / SIZE));
				const uint idx = structIdx + cellIdx;

				double3 localPos = getLocalPos(cellIdx);
				int localX = (int)localPos.x;
				int localZ = (int)localPos.z;

				int localIdx = localZ * SIZE + localX;
				us[localIdx] += uPtr[idx];
				thetas[localIdx] += thetaPtr[idx];

				counts[localIdx]++;
			}

			for (uint i = 0; i < 4; ++i)
			{
				double3 u = us[i] / counts[i];
				double norm = u.x * u.x + u.y * u.y + u.z * u.z;

				u = U_INTENSITY * u;
				double theta = THETA_INTENSITY * sqrt(norm) * thetas[i] / counts[i] / PI;

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
			double3 us[4] = {
				double3{ 0, 0, 0 },
				double3{ 0, 0, 0 },
				double3{ 0, 0, 0 },
				double3{ 0, 0, 0 }
			};
			double thetas[4] = { 0, 0, 0, 0 };
			uint counts[4] = { 0, 0, 0, 0 };

			for (uint cellIdx = 0; cellIdx < VALUES_IN_BLOCK; ++cellIdx)
			{
				const uint structIdx = VALUES_IN_BLOCK * ((z / SIZE) * xSize * ySize + (y / SIZE) * xSize + (x / SIZE));
				const uint idx = structIdx + cellIdx;

				double3 localPos = getLocalPos(cellIdx);
				int localX = (int)localPos.x;
				int localY = (int)localPos.y;

				int localIdx = localY * SIZE + localX;
				us[localIdx] += uPtr[idx];
				thetas[localIdx] += thetaPtr[idx];

				counts[localIdx]++;
			}

			for (uint i = 0; i < 4; ++i)
			{
				double3 u = us[i] / counts[i];
				double norm = u.x * u.x + u.y * u.y + u.z * u.z;

				u = U_INTENSITY * u;
				double theta = THETA_INTENSITY * sqrt(norm) * thetas[i] / counts[i] / PI;

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

constexpr double DENSITY_THRESHOLD = 0.0001;
constexpr double DISTANCE_THRESHOLD = 4;

void saveVolume(const std::string& namePrefix, BlockPsis* pPsi, double3* pLocalAvgSpin, double3* pu, double* pTheta, size_t bsize, size_t dxsize, size_t dysize, size_t dzsize, uint iter, double block_scale, double3 p0, double t, const std::string& folder)
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
					double3 localPos = getLocalPos(dualNode);
					double3 doubleGlobalPos = { p0.x + block_scale * (x * BLOCK_WIDTH_X + localPos.x),
						p0.y + block_scale * (y * BLOCK_WIDTH_Y + localPos.y),
						p0.z + block_scale * (z * BLOCK_WIDTH_Z + localPos.z) };
					float3 globalPos = float3{ (float)doubleGlobalPos.x, (float)doubleGlobalPos.y, (float)doubleGlobalPos.z };

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
					double norm_s1 = pPsi[idx].values[dualNode].s1.x * pPsi[idx].values[dualNode].s1.x + pPsi[idx].values[dualNode].s1.y * pPsi[idx].values[dualNode].s1.y;
					double norm_s0 = pPsi[idx].values[dualNode].s0.x * pPsi[idx].values[dualNode].s0.x + pPsi[idx].values[dualNode].s0.y * pPsi[idx].values[dualNode].s0.y;
					double norm_s_1 = pPsi[idx].values[dualNode].s_1.x * pPsi[idx].values[dualNode].s_1.x + pPsi[idx].values[dualNode].s_1.y * pPsi[idx].values[dualNode].s_1.y;

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
	//				double norm_s0 = pPsi[idx].values[dualNode].s0.x * pPsi[idx].values[dualNode].s0.x + pPsi[idx].values[dualNode].s0.y * pPsi[idx].values[dualNode].s0.y;
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
	//				double norm_s_1 = pPsi[idx].values[dualNode].s_1.x * pPsi[idx].values[dualNode].s_1.x + pPsi[idx].values[dualNode].s_1.y * pPsi[idx].values[dualNode].s_1.y;
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
					double3 avgLocalSpin = { 0, 0, 0 };
					double normSq_s1 = pPsi[psiIdx].values[dualNode].s1.x * pPsi[psiIdx].values[dualNode].s1.x + pPsi[psiIdx].values[dualNode].s1.y * pPsi[psiIdx].values[dualNode].s1.y;
					double normSq_s0 = pPsi[psiIdx].values[dualNode].s0.x * pPsi[psiIdx].values[dualNode].s0.x + pPsi[psiIdx].values[dualNode].s0.y * pPsi[psiIdx].values[dualNode].s0.y;
					double normSq_s_1 = pPsi[psiIdx].values[dualNode].s_1.x * pPsi[psiIdx].values[dualNode].s_1.x + pPsi[psiIdx].values[dualNode].s_1.y * pPsi[psiIdx].values[dualNode].s_1.y;
					double density = normSq_s1 + normSq_s0 + normSq_s_1;

					if ((z > 0) && (y > 0) && (x > 0) &&
						(z < dzsize - 1) && (y < dysize - 1) && (x < dxsize - 1))
					{
						const size_t idx = VALUES_IN_BLOCK * ((z - 1) * xStride * yStride + (y - 1) * xStride + (x - 1)) + dualNode;
						avgLocalSpin = pLocalAvgSpin[idx];
					}

					float spinNorm = 0;
					double3 localPos = getLocalPos(dualNode);
					double3 globalPos = { p0.x + block_scale * (x * BLOCK_WIDTH_X + localPos.x),
						p0.y + block_scale * (y * BLOCK_WIDTH_Y + localPos.y),
						p0.z + block_scale * (z * BLOCK_WIDTH_Z + localPos.z) };
					double distance = sqrt(globalPos.x * globalPos.x + globalPos.y * globalPos.y + globalPos.z * globalPos.z);
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
					double normSq_s1 = pPsi[psiIdx].values[dualNode].s1.x * pPsi[psiIdx].values[dualNode].s1.x + pPsi[psiIdx].values[dualNode].s1.y * pPsi[psiIdx].values[dualNode].s1.y;
					double normSq_s0 = pPsi[psiIdx].values[dualNode].s0.x * pPsi[psiIdx].values[dualNode].s0.x + pPsi[psiIdx].values[dualNode].s0.y * pPsi[psiIdx].values[dualNode].s0.y;
					double normSq_s_1 = pPsi[psiIdx].values[dualNode].s_1.x * pPsi[psiIdx].values[dualNode].s_1.x + pPsi[psiIdx].values[dualNode].s_1.y * pPsi[psiIdx].values[dualNode].s_1.y;
					double density = normSq_s1 + normSq_s0 + normSq_s_1;

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
					double normSq_s1 = pPsi[psiIdx].values[dualNode].s1.x * pPsi[psiIdx].values[dualNode].s1.x + pPsi[psiIdx].values[dualNode].s1.y * pPsi[psiIdx].values[dualNode].s1.y;
					double normSq_s0 = pPsi[psiIdx].values[dualNode].s0.x * pPsi[psiIdx].values[dualNode].s0.x + pPsi[psiIdx].values[dualNode].s0.y * pPsi[psiIdx].values[dualNode].s0.y;
					double normSq_s_1 = pPsi[psiIdx].values[dualNode].s_1.x * pPsi[psiIdx].values[dualNode].s_1.x + pPsi[psiIdx].values[dualNode].s_1.y * pPsi[psiIdx].values[dualNode].s_1.y;
					double density = normSq_s1 + normSq_s0 + normSq_s_1;

					float sx = 0;
					float sy = 0;
					float sz = 0;

					if ((density > DENSITY_THRESHOLD) && (z > 0) && (y > 0) && (x > 0) &&
						(z < dzsize - 1) && (y < dysize - 1) && (x < dxsize - 1))
					{
						const size_t idx = VALUES_IN_BLOCK * ((z - 1) * xStride * yStride + (y - 1) * xStride + (x - 1)) + dualNode;
						double3 avgLocalSpin = pLocalAvgSpin[idx];

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
					double normSq_s1 = pPsi[psiIdx].values[dualNode].s1.x * pPsi[psiIdx].values[dualNode].s1.x + pPsi[psiIdx].values[dualNode].s1.y * pPsi[psiIdx].values[dualNode].s1.y;
					double normSq_s0 = pPsi[psiIdx].values[dualNode].s0.x * pPsi[psiIdx].values[dualNode].s0.x + pPsi[psiIdx].values[dualNode].s0.y * pPsi[psiIdx].values[dualNode].s0.y;
					double normSq_s_1 = pPsi[psiIdx].values[dualNode].s_1.x * pPsi[psiIdx].values[dualNode].s_1.x + pPsi[psiIdx].values[dualNode].s_1.y * pPsi[psiIdx].values[dualNode].s_1.y;
					double density = normSq_s1 + normSq_s0 + normSq_s_1;

					float ux = 0;
					float uy = 0;
					float uz = 0;

					if ((density > DENSITY_THRESHOLD) && (z > 0) && (y > 0) && (x > 0) &&
						(z < dzsize - 1) && (y < dysize - 1) && (x < dxsize - 1))
					{
						const size_t idx = VALUES_IN_BLOCK * ((z - 1) * xStride * yStride + (y - 1) * xStride + (x - 1)) + dualNode;
						double3 u = pu[idx];

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

double3 centerOfMass(BlockPsis* h_evenPsi, size_t bsize, size_t dxsize, size_t dysize, size_t dzsize, double block_scale, double3 p0)
{
	double3 com{};

	double totDens = 0;

	for (uint z = 0; z < dzsize; ++z)
	{
		for (uint x = 0; x < dxsize; ++x)
		{
			for (uint y = 0; y < dysize; ++y)
			{
				const uint idx = z * dxsize * dysize + y * dxsize + x;
				for (uint dualNode = 0; dualNode < VALUES_IN_BLOCK; ++dualNode)
				{
					double3 localPos = getLocalPos(dualNode);
					double3 globalPos = { p0.x + block_scale * ((x - 1.0) * BLOCK_WIDTH_X + localPos.x),
										  p0.y + block_scale * ((y - 1.0) * BLOCK_WIDTH_Y + localPos.y),
										  p0.z + block_scale * ((z - 1.0) * BLOCK_WIDTH_Z + localPos.z) };

					double normSq_s1 = h_evenPsi[idx].values[dualNode].s1.x * h_evenPsi[idx].values[dualNode].s1.x + h_evenPsi[idx].values[dualNode].s1.y * h_evenPsi[idx].values[dualNode].s1.y;
					double normSq_s0 = h_evenPsi[idx].values[dualNode].s0.x * h_evenPsi[idx].values[dualNode].s0.x + h_evenPsi[idx].values[dualNode].s0.y * h_evenPsi[idx].values[dualNode].s0.y;
					double normSq_s_1 = h_evenPsi[idx].values[dualNode].s_1.x * h_evenPsi[idx].values[dualNode].s_1.x + h_evenPsi[idx].values[dualNode].s_1.y * h_evenPsi[idx].values[dualNode].s_1.y;
					double density = normSq_s1 + normSq_s0 + normSq_s_1;

					com += density * globalPos;
					totDens += density;
				}
			}
		}
	}

	return com / totDens;
}

#endif // UTILS