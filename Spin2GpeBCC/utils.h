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
__host__ __device__ __inline__ double2 operator*(double2 a, double b)
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

struct Complex5Vec
{
	double2 s2;
	double2 s1;
	double2 s0;
	double2 s_1;
	double2 s_2;
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

void printBasis()
{
	std::cout << "Using knot/skyrmion creation process" << std::endl;
#if BASIS == Z_QUANTIZED
	std::cout << "Using z-quantized basis!" << std::endl;
#elif BASIS == Y_QUANTIZED
	std::cout << "Using y-quantized basis!" << std::endl;
#elif BASIS == X_QUANTIZED
	std::cout << "Using x-quantized basis!" << std::endl;
#endif
}

void drawIandR(const std::string& name, BlockPsis* h_evenPsi, size_t dxsize, size_t dysize, size_t dzsize, double t, MagFields Bs, const double3 p0, double block_scale)
{
	const int SIZE = 2;
	const double INTENSITY = 1;
	const double MAG_ZERO = 0.195;
	const int width = dxsize * SIZE, height = dysize * SIZE, depth = dzsize * SIZE;
	Picture pic1(width * 5, height * 2);

	// XZ-plane
	for (uint k = 0; k < depth; ++k)
	{
		for (uint i = 0; i < width; i++)
		{
			double2 norm_s2 =  {0, 0};
			double2 norm_s1 =  {0, 0};
			double2 norm_s0 =  {0, 0};
			double2 norm_s_1 = {0, 0};
			double2 norm_s_2 = {0, 0};
			double minB = 99999999999999.9;
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
						double3 localPos = getLocalPos(dualNode);
						const double3 globalPos = { p0.x + block_scale * (((i - 1) / SIZE) * BLOCK_WIDTH_X + localPos.x),
													p0.y + block_scale * (((j - 1) / SIZE) * BLOCK_WIDTH_Y + localPos.y),
													p0.z + block_scale * (((k - 1) / SIZE) * BLOCK_WIDTH_Z + localPos.z) };

						//double3 B = magneticField(globalPos, Bs.Bq, Bs.Bz);
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
				const double2 s2 = INTENSITY * norm_s2;
				const double2 s1 = INTENSITY * norm_s1;
				const double2 s0 = INTENSITY * norm_s0;
				const double2 s_1 = INTENSITY * norm_s_1;
				const double2 s_2 = INTENSITY * norm_s_2;
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
			double2 norm_s2 = { 0, 0 };
			double2 norm_s1 = { 0, 0 };
			double2 norm_s0 = { 0, 0 };
			double2 norm_s_1 = { 0, 0 };
			double2 norm_s_2 = { 0, 0 };
			double minB = 99999999999999.9;
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
						double3 localPos = getLocalPos(dualNode);
						const double3 globalPos = { p0.x + block_scale * (((i - 1) / SIZE) * BLOCK_WIDTH_X + localPos.x),
													p0.y + block_scale * (((j - 1) / SIZE) * BLOCK_WIDTH_Y + localPos.y),
													p0.z + block_scale * (((k - 1) / SIZE) * BLOCK_WIDTH_Z + localPos.z) };

						//double3 B = magneticField(globalPos, Bs.Bq, Bs.Bz);
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
				const double2 s2 = INTENSITY * norm_s2;
				const double2 s1 = INTENSITY * norm_s1;
				const double2 s0 = INTENSITY * norm_s0;
				const double2 s_1 = INTENSITY * norm_s_1;
				const double2 s_2 = INTENSITY * norm_s_2;

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

	pic1.save("results/" + name + toString(t) + "ms.bmp", false);
	//pic1.save("mag_pos.bmp", false);
}

void drawDensity(const std::string& name, BlockPsis* h_evenPsi, size_t dxsize, size_t dysize, size_t dzsize, double t, MagFields Bs, const double3 p0, double block_scale)
{
	const int SIZE = 2;
	const double INTENSITY = 1;
	const double MAG_ZERO = 0.195;
	const int width = dxsize * SIZE, height = dysize * SIZE, depth = dzsize * SIZE;
	Picture pic1(width * 5, height * 3);

	// XZ-plane
	for (uint k = 0; k < depth; ++k)
	{
		for (uint i = 0; i < width; i++)
		{
			double norm_s2 = 0;
			double norm_s1 = 0;
			double norm_s0 = 0;
			double norm_s_1 = 0;
			double norm_s_2 = 0;
			double minB = 99999999999999.9;
			for (uint j = 0; j < height; j++)
			{
				const uint idx = (k / SIZE) * dxsize * dysize + (j / SIZE) * dxsize + i / SIZE;
				for (uint dualNode = 0; dualNode < VALUES_IN_BLOCK; ++dualNode)
				{
					double2 s2  = h_evenPsi[idx].values[dualNode].s2;
					double2 s1  = h_evenPsi[idx].values[dualNode].s1;
					double2 s0  = h_evenPsi[idx].values[dualNode].s0;
					double2 s_1 = h_evenPsi[idx].values[dualNode].s_1;
					double2 s_2 = h_evenPsi[idx].values[dualNode].s_2;

#if BASIS == X_QUANTIZED
					double c = sqrt(6) * 0.25;
					double2 x_s2  = 0.25 * s2 + 0.5 * s1 +   c * s0 + 0.5 * s_1 + 0.25 * s_2;
					double2 x_s1  = -0.5 * s2 - 0.5 * s1            + 0.5 * s_1 +  0.5 * s_2;
					double2 x_s0  =    c * s2            - 0.5 * s0             +    c * s_2;
					double2 x_s_1 = -0.5 * s2 + 0.5 * s1            - 0.5 * s_1 +  0.5 * s_2;
					double2 x_s_2 = 0.25 * s2 - 0.5 * s1 +   c * s0 - 0.5 * s_1 + 0.25 * s_2;

					s2 =  x_s2;
					s1 =  x_s1;
					s0 =  x_s0;
					s_1 = x_s_1;
					s_2 = x_s_2;
#elif BASIS == Y_QUANTIZED
					double c = sqrt(6) * 0.25;
					double2 im = { 0, 1 };
					double2 y_s2  =      0.25 * s2 - im * 0.5 * s1 -   c * s0 + im * 0.5 * s_1 +     0.25 * s_2;
					double2 y_s1  = -im * 0.5 * s2 -      0.5 * s1            -      0.5 * s_1 + im * 0.5 * s_2;
					double2 y_s0  =        -c * s2                 - 0.5 * s0                  -        c * s_2;
					double2 y_s_1 =  im * 0.5 * s2 -      0.5 * s1            -      0.5 * s_1 - im * 0.5 * s_2;
					double2 y_s_2 =      0.25 * s2 + im * 0.5 * s1 -   c * s0 - im * 0.5 * s_1 +     0.25 * s_2;

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
						double3 localPos = getLocalPos(dualNode);
						const double3 globalPos = { p0.x + block_scale * (((i - 1) / SIZE) * BLOCK_WIDTH_X + localPos.x),
													p0.y + block_scale * (((j - 1) / SIZE) * BLOCK_WIDTH_Y + localPos.y),
													p0.z + block_scale * (((k - 1) / SIZE) * BLOCK_WIDTH_Z + localPos.z) };

						//double3 B = magneticField(globalPos, Bs.Bq, Bs.Bz);
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
				const double s2 = INTENSITY * norm_s2;
				const double s1 = INTENSITY * norm_s1;
				const double s0 = INTENSITY * norm_s0;
				const double s_1 = INTENSITY * norm_s_1;
				const double s_2 = INTENSITY * norm_s_2;
				pic1.setColor(i,             k, Vector4(s2, s2, s2, 1.0));
				pic1.setColor(width + i,     k, Vector4(s1, s1, s1, 1.0));
				pic1.setColor(2 * width + i, k, Vector4(s0, s0, s0, 1.0));
				pic1.setColor(3 * width + i, k, Vector4(s_1, s_1, s_1, 1.0));
				pic1.setColor(4 * width + i, k, Vector4(s_2, s_2, s_2, 1.0));
			}
		}
	}

	// YZ-plane
	for (uint k = 0; k < depth; ++k)
	{
		for (uint j = 0; j < height; j++)
		{
			double norm_s2 = 0;
			double norm_s1 = 0;
			double norm_s0 = 0;
			double norm_s_1 = 0;
			double norm_s_2 = 0;
			double minB = 99999999999999.9;
			for (uint i = 0; i < width; i++)
			{
				const uint idx = (k / SIZE) * dxsize * dysize + (j / SIZE) * dxsize + i / SIZE;
				for (uint dualNode = 0; dualNode < VALUES_IN_BLOCK; ++dualNode)
				{
					double2 s2 = h_evenPsi[idx].values[dualNode].s2;
					double2 s1 = h_evenPsi[idx].values[dualNode].s1;
					double2 s0 = h_evenPsi[idx].values[dualNode].s0;
					double2 s_1 = h_evenPsi[idx].values[dualNode].s_1;
					double2 s_2 = h_evenPsi[idx].values[dualNode].s_2;

#if BASIS == X_QUANTIZED
					double c = sqrt(6) * 0.25;
					double2 x_s2  = 0.25 * s2 + 0.5 * s1 +   c * s0 + 0.5 * s_1 + 0.25 * s_2;
					double2 x_s1  = -0.5 * s2 - 0.5 * s1            + 0.5 * s_1 +  0.5 * s_2;
					double2 x_s0  =    c * s2            - 0.5 * s0             +    c * s_2;
					double2 x_s_1 = -0.5 * s2 + 0.5 * s1            - 0.5 * s_1 +  0.5 * s_2;
					double2 x_s_2 = 0.25 * s2 - 0.5 * s1 +   c * s0 - 0.5 * s_1 + 0.25 * s_2;

					s2 = x_s2;
					s1 = x_s1;
					s0 = x_s0;
					s_1 = x_s_1;
					s_2 = x_s_2;
#elif BASIS == Y_QUANTIZED
					double c = sqrt(6) * 0.25;
					double2 im = { 0, 1 };
					double2 y_s2  =      0.25 * s2 - im * 0.5 * s1 -   c * s0 + im * 0.5 * s_1 +     0.25 * s_2;
					double2 y_s1  = -im * 0.5 * s2 -      0.5 * s1            -      0.5 * s_1 + im * 0.5 * s_2;
					double2 y_s0  =        -c * s2                 - 0.5 * s0                  -        c * s_2;
					double2 y_s_1 =  im * 0.5 * s2 -      0.5 * s1            -      0.5 * s_1 - im * 0.5 * s_2;
					double2 y_s_2 =      0.25 * s2 + im * 0.5 * s1 -   c * s0 - im * 0.5 * s_1 +     0.25 * s_2;

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
						double3 localPos = getLocalPos(dualNode);
						const double3 globalPos = { p0.x + block_scale * (((i - 1) / SIZE) * BLOCK_WIDTH_X + localPos.x),
													p0.y + block_scale * (((j - 1) / SIZE) * BLOCK_WIDTH_Y + localPos.y),
													p0.z + block_scale * (((k - 1) / SIZE) * BLOCK_WIDTH_Z + localPos.z) };

						//double3 B = magneticField(globalPos, Bs.Bq, Bs.Bz);
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
				const double s2 = INTENSITY * norm_s2;
				const double s1 = INTENSITY * norm_s1;
				const double s0 = INTENSITY * norm_s0;
				const double s_1 = INTENSITY * norm_s_1;
				const double s_2 = INTENSITY * norm_s_2;
				pic1.setColor(j,             height + k, Vector4(s2, s2, s2, 1.0));
				pic1.setColor(width + j,     height + k, Vector4(s1, s1, s1, 1.0));
				pic1.setColor(2 * width + j, height + k, Vector4(s0, s0, s0, 1.0));
				pic1.setColor(3 * width + j, height + k, Vector4(s_1, s_1, s_1, 1.0));
				pic1.setColor(4 * width + j, height + k, Vector4(s_2, s_2, s_2, 1.0));
			}
		}
	}
	
	// XY-plane
	for (uint j = 0; j < height; j++)
	{
		for (uint i = 0; i < width; i++)
		{
			double norm_s2 = 0;
			double norm_s1 = 0;
			double norm_s0 = 0;
			double norm_s_1 = 0;
			double norm_s_2 = 0;
			double minB = 99999999999999.9;
			for (uint k = 0; k < depth; ++k)
			{
				const uint idx = (k / SIZE) * dxsize * dysize + (j / SIZE) * dxsize + i / SIZE;
				for (uint dualNode = 0; dualNode < VALUES_IN_BLOCK; ++dualNode)
				{
					double2 s2 = h_evenPsi[idx].values[dualNode].s2;
					double2 s1 = h_evenPsi[idx].values[dualNode].s1;
					double2 s0 = h_evenPsi[idx].values[dualNode].s0;
					double2 s_1 = h_evenPsi[idx].values[dualNode].s_1;
					double2 s_2 = h_evenPsi[idx].values[dualNode].s_2;

#if BASIS == X_QUANTIZED
					double c = sqrt(6) * 0.25;
					double2 x_s2  = 0.25 * s2 + 0.5 * s1 +   c * s0 + 0.5 * s_1 + 0.25 * s_2;
					double2 x_s1  = -0.5 * s2 - 0.5 * s1            + 0.5 * s_1 +  0.5 * s_2;
					double2 x_s0  =    c * s2            - 0.5 * s0             +    c * s_2;
					double2 x_s_1 = -0.5 * s2 + 0.5 * s1            - 0.5 * s_1 +  0.5 * s_2;
					double2 x_s_2 = 0.25 * s2 - 0.5 * s1 +   c * s0 - 0.5 * s_1 + 0.25 * s_2;

					s2 = x_s2;
					s1 = x_s1;
					s0 = x_s0;
					s_1 = x_s_1;
					s_2 = x_s_2;
#elif BASIS == Y_QUANTIZED
					double c = sqrt(6) * 0.25;
					double2 im = { 0, 1 };
					double2 y_s2  =      0.25 * s2 - im * 0.5 * s1 -   c * s0 + im * 0.5 * s_1 +     0.25 * s_2;
					double2 y_s1  = -im * 0.5 * s2 -      0.5 * s1            -      0.5 * s_1 + im * 0.5 * s_2;
					double2 y_s0  =        -c * s2                 - 0.5 * s0                  -        c * s_2;
					double2 y_s_1 =  im * 0.5 * s2 -      0.5 * s1            -      0.5 * s_1 - im * 0.5 * s_2;
					double2 y_s_2 =      0.25 * s2 + im * 0.5 * s1 -   c * s0 - im * 0.5 * s_1 +     0.25 * s_2;

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
						double3 localPos = getLocalPos(dualNode);
						const double3 globalPos = { p0.x + block_scale * (((i - 1) / SIZE) * BLOCK_WIDTH_X + localPos.x),
													p0.y + block_scale * (((j - 1) / SIZE) * BLOCK_WIDTH_Y + localPos.y),
													p0.z + block_scale * (((k - 1) / SIZE) * BLOCK_WIDTH_Z + localPos.z) };

						//double3 B = magneticField(globalPos, Bs.Bq, Bs.Bz);
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
				const double s2 = INTENSITY * norm_s2;
				const double s1 = INTENSITY * norm_s1;
				const double s0 = INTENSITY * norm_s0;
				const double s_1 = INTENSITY * norm_s_1;
				const double s_2 = INTENSITY * norm_s_2;

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

	pic1.save("results/" + name + toString(t) + "ms.bmp", false);
	//pic1.save("mag_pos.bmp", false);
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

void saveVolume(const std::string& namePrefix, BlockPsis* pPsi, size_t bsize, size_t dxsize, size_t dysize, size_t dzsize, double block_scale, double3 p0, double t)
{
	std::ofstream file;
	file.open("vtks/" + namePrefix + std::to_string(t) + ".vtk", std::ios::out | std::ios::binary);

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
	file << "SCALARS m=2 float 1" << std::endl;
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
					double2 s2 =  pPsi[idx].values[dualNode].s2;
					double2 s1 =  pPsi[idx].values[dualNode].s1;
					double2 s0 =  pPsi[idx].values[dualNode].s0;
					double2 s_1 = pPsi[idx].values[dualNode].s_1;
					double2 s_2 = pPsi[idx].values[dualNode].s_2;
#if BASIS == X_QUANTIZED
					double c = sqrt(6) * 0.25;
					double2 x_s2  = 0.25 * s2 + 0.5 * s1 +   c * s0 + 0.5 * s_1 + 0.25 * s_2;

					s2 = x_s2;
#elif BASIS == Y_QUANTIZED
					double c = sqrt(6) * 0.25;
					double2 im = { 0, 1 };
					double2 y_s2  =      0.25 * s2 - im * 0.5 * s1 -   c * s0 + im * 0.5 * s_1 +     0.25 * s_2;

					s2 = y_s2;
#endif

					double dens_m2 = s2.x * s2.x + s2.y * s2.y;
	
					float density = (float)(dens_m2);
					swapEnd(density);
					file.write((char*)&density, sizeof(float));
				}
			}
		}
	}
	
	file << std::endl << "SCALARS m=1 float 1" << std::endl;
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
					double2 s2 = pPsi[idx].values[dualNode].s2;
					double2 s1 = pPsi[idx].values[dualNode].s1;
					double2 s0 = pPsi[idx].values[dualNode].s0;
					double2 s_1 = pPsi[idx].values[dualNode].s_1;
					double2 s_2 = pPsi[idx].values[dualNode].s_2;
#if BASIS == X_QUANTIZED
					double c = sqrt(6) * 0.25;

					double2 x_s1 = -0.5 * s2 - 0.5 * s1 + 0.5 * s_1 + 0.5 * s_2;


					s1 = x_s1;
#elif BASIS == Y_QUANTIZED
					double c = sqrt(6) * 0.25;
					double2 im = { 0, 1 };

					double2 y_s1 = -im * 0.5 * s2 - 0.5 * s1 - 0.5 * s_1 + im * 0.5 * s_2;

					s1 = y_s1;
#endif

					double dens_m1 = s1.x * s1.x + s1.y * s1.y;
	
					float density = (float)(dens_m1);
					swapEnd(density);
					file.write((char*)&density, sizeof(float));
				}
			}
		}
	}

	file << std::endl << "SCALARS m=0 float 1" << std::endl;
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
					double2 s2 = pPsi[idx].values[dualNode].s2;
					double2 s1 = pPsi[idx].values[dualNode].s1;
					double2 s0 = pPsi[idx].values[dualNode].s0;
					double2 s_1 = pPsi[idx].values[dualNode].s_1;
					double2 s_2 = pPsi[idx].values[dualNode].s_2;
#if BASIS == X_QUANTIZED
					double c = sqrt(6) * 0.25;

					double2 x_s0 = c * s2 - 0.5 * s0 + c * s_2;

					s0 = x_s0;
#elif BASIS == Y_QUANTIZED
					double c = sqrt(6) * 0.25;
					double2 im = { 0, 1 };

					double2 y_s0 = -c * s2 - 0.5 * s0 - c * s_2;

					s0 = y_s0;
#endif

					double dens_m0 = s0.x * s0.x + s0.y * s0.y;

					float density = (float)(dens_m0);
					swapEnd(density);
					file.write((char*)&density, sizeof(float));
				}
			}
		}
	}

	file << std::endl << "SCALARS m=-1 float 1" << std::endl;
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
					double2 s2 = pPsi[idx].values[dualNode].s2;
					double2 s1 = pPsi[idx].values[dualNode].s1;
					double2 s0 = pPsi[idx].values[dualNode].s0;
					double2 s_1 = pPsi[idx].values[dualNode].s_1;
					double2 s_2 = pPsi[idx].values[dualNode].s_2;
#if BASIS == X_QUANTIZED
					double c = sqrt(6) * 0.25;

					double2 x_s_1 = -0.5 * s2 + 0.5 * s1 - 0.5 * s_1 + 0.5 * s_2;

					s_1 = x_s_1;
#elif BASIS == Y_QUANTIZED
					double c = sqrt(6) * 0.25;
					double2 im = { 0, 1 };

					double2 y_s_1 = im * 0.5 * s2 - 0.5 * s1 - 0.5 * s_1 - im * 0.5 * s_2;

					s_1 = y_s_1;
#endif

					double dens_m_1 = s_1.x * s_1.x + s_1.y * s_1.y;

					float density = (float)(dens_m_1);
					swapEnd(density);
					file.write((char*)&density, sizeof(float));
				}
			}
		}
	}

	file << std::endl << "SCALARS m=-2 float 1" << std::endl;
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
					double2 s2 = pPsi[idx].values[dualNode].s2;
					double2 s1 = pPsi[idx].values[dualNode].s1;
					double2 s0 = pPsi[idx].values[dualNode].s0;
					double2 s_1 = pPsi[idx].values[dualNode].s_1;
					double2 s_2 = pPsi[idx].values[dualNode].s_2;
#if BASIS == X_QUANTIZED
					double c = sqrt(6) * 0.25;

					double2 x_s_2 = 0.25 * s2 - 0.5 * s1 + c * s0 - 0.5 * s_1 + 0.25 * s_2;

					s_2 = x_s_2;
#elif BASIS == Y_QUANTIZED
					double c = sqrt(6) * 0.25;
					double2 im = { 0, 1 };

					double2 y_s_2 = 0.25 * s2 + im * 0.5 * s1 - c * s0 - im * 0.5 * s_1 + 0.25 * s_2;

					s_2 = y_s_2;
#endif

					double dens_m_2 = s_2.x * s_2.x + s_2.y * s_2.y;

					float density = (float)(dens_m_2);
					swapEnd(density);
					file.write((char*)&density, sizeof(float));
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

void saveSpinor(const std::string& namePrefix, BlockPsis* pPsi, size_t bsize, size_t dxsize, size_t dysize, size_t dzsize, double block_scale, double3 p0, double t)
{
	std::ofstream file;
	file.open("spinor_vtks/" + namePrefix + std::to_string(t) + ".vtk", std::ios::out | std::ios::binary);

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
	file << "SCALARS r_m=2 float 1" << std::endl;
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
					double2 s2 = pPsi[idx].values[dualNode].s2;
					double2 s1 = pPsi[idx].values[dualNode].s1;
					double2 s0 = pPsi[idx].values[dualNode].s0;
					double2 s_1 = pPsi[idx].values[dualNode].s_1;
					double2 s_2 = pPsi[idx].values[dualNode].s_2;

#if BASIS == X_QUANTIZED
					double c = sqrt(6) * 0.25;
					double2 x_s2  = 0.25 * s2 + 0.5 * s1 +   c * s0 + 0.5 * s_1 + 0.25 * s_2;
					double2 x_s1  = -0.5 * s2 - 0.5 * s1            + 0.5 * s_1 +  0.5 * s_2;
					double2 x_s0  =    c * s2            - 0.5 * s0             +    c * s_2;
					double2 x_s_1 = -0.5 * s2 + 0.5 * s1            - 0.5 * s_1 +  0.5 * s_2;
					double2 x_s_2 = 0.25 * s2 - 0.5 * s1 +   c * s0 - 0.5 * s_1 + 0.25 * s_2;

					s2 = x_s2;
					s1 = x_s1;
					s0 = x_s0;
					s_1 = x_s_1;
					s_2 = x_s_2;
#elif BASIS == Y_QUANTIZED
					double c = sqrt(6) * 0.25;
					double2 im = { 0, 1 };
					double2 y_s2  =      0.25 * s2 - im * 0.5 * s1 -   c * s0 + im * 0.5 * s_1 +     0.25 * s_2;
					double2 y_s1  = -im * 0.5 * s2 -      0.5 * s1            -      0.5 * s_1 + im * 0.5 * s_2;
					double2 y_s0  =        -c * s2                 - 0.5 * s0                  -        c * s_2;
					double2 y_s_1 =  im * 0.5 * s2 -      0.5 * s1            -      0.5 * s_1 - im * 0.5 * s_2;
					double2 y_s_2 =      0.25 * s2 + im * 0.5 * s1 -   c * s0 - im * 0.5 * s_1 +     0.25 * s_2;

					s2 = y_s2;
					s1 = y_s1;
					s0 = y_s0;
					s_1 = y_s_1;
					s_2 = y_s_2;
#endif

					double dens_s2 = s2.x * s2.x + s2.y * s2.y;
					double dens_s1 = s1.x * s1.x + s1.y * s1.y;
					double dens_s0 = s0.x * s0.x + s0.y * s0.y;
					double dens_s_1 = s_1.x * s_1.x + s_1.y * s_1.y;
					double dens_s_2 = s_2.x * s_2.x + s_2.y * s_2.y;
					double dens = dens_s2 + dens_s1 + dens_s0 + dens_s_1 + dens_s_2;
					
					float s2_r = 0;
					if (DENSITY_THRESHOLD < dens)
						s2_r = (float)(s2.x / sqrt(dens));
					swapEnd(s2_r);
					file.write((char*)&s2_r, sizeof(float));
				}
			}
		}
	}
	file << std::endl << "SCALARS i_m=2 float 1" << std::endl;
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
					double2 s2 = pPsi[idx].values[dualNode].s2;
					double2 s1 = pPsi[idx].values[dualNode].s1;
					double2 s0 = pPsi[idx].values[dualNode].s0;
					double2 s_1 = pPsi[idx].values[dualNode].s_1;
					double2 s_2 = pPsi[idx].values[dualNode].s_2;
#if BASIS == X_QUANTIZED
					double c = sqrt(6) * 0.25;
					double2 x_s2  = 0.25 * s2 + 0.5 * s1 +   c * s0 + 0.5 * s_1 + 0.25 * s_2;
					double2 x_s1  = -0.5 * s2 - 0.5 * s1            + 0.5 * s_1 +  0.5 * s_2;
					double2 x_s0  =    c * s2            - 0.5 * s0             +    c * s_2;
					double2 x_s_1 = -0.5 * s2 + 0.5 * s1            - 0.5 * s_1 +  0.5 * s_2;
					double2 x_s_2 = 0.25 * s2 - 0.5 * s1 +   c * s0 - 0.5 * s_1 + 0.25 * s_2;

					s2 = x_s2;
					s1 = x_s1;
					s0 = x_s0;
					s_1 = x_s_1;
					s_2 = x_s_2;
#elif BASIS == Y_QUANTIZED
					double c = sqrt(6) * 0.25;
					double2 im = { 0, 1 };
					double2 y_s2  =      0.25 * s2 - im * 0.5 * s1 -   c * s0 + im * 0.5 * s_1 +     0.25 * s_2;
					double2 y_s1  = -im * 0.5 * s2 -      0.5 * s1            -      0.5 * s_1 + im * 0.5 * s_2;
					double2 y_s0  =        -c * s2                 - 0.5 * s0                  -        c * s_2;
					double2 y_s_1 =  im * 0.5 * s2 -      0.5 * s1            -      0.5 * s_1 - im * 0.5 * s_2;
					double2 y_s_2 =      0.25 * s2 + im * 0.5 * s1 -   c * s0 - im * 0.5 * s_1 +     0.25 * s_2;

					s2 = y_s2;
					s1 = y_s1;
					s0 = y_s0;
					s_1 = y_s_1;
					s_2 = y_s_2;
#endif

					double dens_s2 = s2.x * s2.x + s2.y * s2.y;
					double dens_s1 = s1.x * s1.x + s1.y * s1.y;
					double dens_s0 = s0.x * s0.x + s0.y * s0.y;
					double dens_s_1 = s_1.x * s_1.x + s_1.y * s_1.y;
					double dens_s_2 = s_2.x * s_2.x + s_2.y * s_2.y;
					double dens = dens_s2 + dens_s1 + dens_s0 + dens_s_1 + dens_s_2;

					float s2_i = 0;
					if (DENSITY_THRESHOLD < dens)
						s2_i = (float)(s2.y / sqrt(dens));
					swapEnd(s2_i);
					file.write((char*)&s2_i, sizeof(float));
				}
			}
		}
	}

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
					double2 s2 = pPsi[idx].values[dualNode].s2;
					double2 s1 = pPsi[idx].values[dualNode].s1;
					double2 s0 = pPsi[idx].values[dualNode].s0;
					double2 s_1 = pPsi[idx].values[dualNode].s_1;
					double2 s_2 = pPsi[idx].values[dualNode].s_2;
#if BASIS == X_QUANTIZED
					double c = sqrt(6) * 0.25;
					double2 x_s2  = 0.25 * s2 + 0.5 * s1 +   c * s0 + 0.5 * s_1 + 0.25 * s_2;
					double2 x_s1  = -0.5 * s2 - 0.5 * s1            + 0.5 * s_1 +  0.5 * s_2;
					double2 x_s0  =    c * s2            - 0.5 * s0             +    c * s_2;
					double2 x_s_1 = -0.5 * s2 + 0.5 * s1            - 0.5 * s_1 +  0.5 * s_2;
					double2 x_s_2 = 0.25 * s2 - 0.5 * s1 +   c * s0 - 0.5 * s_1 + 0.25 * s_2;

					s2 = x_s2;
					s1 = x_s1;
					s0 = x_s0;
					s_1 = x_s_1;
					s_2 = x_s_2;
#elif BASIS == Y_QUANTIZED
					double c = sqrt(6) * 0.25;
					double2 im = { 0, 1 };
					double2 y_s2  =      0.25 * s2 - im * 0.5 * s1 -   c * s0 + im * 0.5 * s_1 +     0.25 * s_2;
					double2 y_s1  = -im * 0.5 * s2 -      0.5 * s1            -      0.5 * s_1 + im * 0.5 * s_2;
					double2 y_s0  =        -c * s2                 - 0.5 * s0                  -        c * s_2;
					double2 y_s_1 =  im * 0.5 * s2 -      0.5 * s1            -      0.5 * s_1 - im * 0.5 * s_2;
					double2 y_s_2 =      0.25 * s2 + im * 0.5 * s1 -   c * s0 - im * 0.5 * s_1 +     0.25 * s_2;

					s2 = y_s2;
					s1 = y_s1;
					s0 = y_s0;
					s_1 = y_s_1;
					s_2 = y_s_2;
#endif

					double dens_s2 = s2.x * s2.x + s2.y * s2.y;
					double dens_s1 = s1.x * s1.x + s1.y * s1.y;
					double dens_s0 = s0.x * s0.x + s0.y * s0.y;
					double dens_s_1 = s_1.x * s_1.x + s_1.y * s_1.y;
					double dens_s_2 = s_2.x * s_2.x + s_2.y * s_2.y;
					double dens = dens_s2 + dens_s1 + dens_s0 + dens_s_1 + dens_s_2;

					float s1_r = 0;
					if (DENSITY_THRESHOLD < dens)
						s1_r = (float)(s1.x / sqrt(dens));
					swapEnd(s1_r);
					file.write((char*)&s1_r, sizeof(float));
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
					double2 s2 = pPsi[idx].values[dualNode].s2;
					double2 s1 = pPsi[idx].values[dualNode].s1;
					double2 s0 = pPsi[idx].values[dualNode].s0;
					double2 s_1 = pPsi[idx].values[dualNode].s_1;
					double2 s_2 = pPsi[idx].values[dualNode].s_2;
#if BASIS == X_QUANTIZED
					double c = sqrt(6) * 0.25;
					double2 x_s2  = 0.25 * s2 + 0.5 * s1 +   c * s0 + 0.5 * s_1 + 0.25 * s_2;
					double2 x_s1  = -0.5 * s2 - 0.5 * s1            + 0.5 * s_1 +  0.5 * s_2;
					double2 x_s0  =    c * s2            - 0.5 * s0             +    c * s_2;
					double2 x_s_1 = -0.5 * s2 + 0.5 * s1            - 0.5 * s_1 +  0.5 * s_2;
					double2 x_s_2 = 0.25 * s2 - 0.5 * s1 +   c * s0 - 0.5 * s_1 + 0.25 * s_2;

					s2 = x_s2;
					s1 = x_s1;
					s0 = x_s0;
					s_1 = x_s_1;
					s_2 = x_s_2;
#elif BASIS == Y_QUANTIZED
					double c = sqrt(6) * 0.25;
					double2 im = { 0, 1 };
					double2 y_s2  =      0.25 * s2 - im * 0.5 * s1 -   c * s0 + im * 0.5 * s_1 +     0.25 * s_2;
					double2 y_s1  = -im * 0.5 * s2 -      0.5 * s1            -      0.5 * s_1 + im * 0.5 * s_2;
					double2 y_s0  =        -c * s2                 - 0.5 * s0                  -        c * s_2;
					double2 y_s_1 =  im * 0.5 * s2 -      0.5 * s1            -      0.5 * s_1 - im * 0.5 * s_2;
					double2 y_s_2 =      0.25 * s2 + im * 0.5 * s1 -   c * s0 - im * 0.5 * s_1 +     0.25 * s_2;

					s2 = y_s2;
					s1 = y_s1;
					s0 = y_s0;
					s_1 = y_s_1;
					s_2 = y_s_2;
#endif

					double dens_s2 = s2.x * s2.x + s2.y * s2.y;
					double dens_s1 = s1.x * s1.x + s1.y * s1.y;
					double dens_s0 = s0.x * s0.x + s0.y * s0.y;
					double dens_s_1 = s_1.x * s_1.x + s_1.y * s_1.y;
					double dens_s_2 = s_2.x * s_2.x + s_2.y * s_2.y;
					double dens = dens_s2 + dens_s1 + dens_s0 + dens_s_1 + dens_s_2;

					float s1_i = 0;
					if (DENSITY_THRESHOLD < dens)
						s1_i = (float)(s1.y / sqrt(dens));
					swapEnd(s1_i);
					file.write((char*)&s1_i, sizeof(float));
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
					double2 s2 = pPsi[idx].values[dualNode].s2;
					double2 s1 = pPsi[idx].values[dualNode].s1;
					double2 s0 = pPsi[idx].values[dualNode].s0;
					double2 s_1 = pPsi[idx].values[dualNode].s_1;
					double2 s_2 = pPsi[idx].values[dualNode].s_2;
#if BASIS == X_QUANTIZED
					double c = sqrt(6) * 0.25;
					double2 x_s2  = 0.25 * s2 + 0.5 * s1 +   c * s0 + 0.5 * s_1 + 0.25 * s_2;
					double2 x_s1  = -0.5 * s2 - 0.5 * s1            + 0.5 * s_1 +  0.5 * s_2;
					double2 x_s0  =    c * s2            - 0.5 * s0             +    c * s_2;
					double2 x_s_1 = -0.5 * s2 + 0.5 * s1            - 0.5 * s_1 +  0.5 * s_2;
					double2 x_s_2 = 0.25 * s2 - 0.5 * s1 +   c * s0 - 0.5 * s_1 + 0.25 * s_2;

					s2 = x_s2;
					s1 = x_s1;
					s0 = x_s0;
					s_1 = x_s_1;
					s_2 = x_s_2;
#elif BASIS == Y_QUANTIZED
					double c = sqrt(6) * 0.25;
					double2 im = { 0, 1 };
					double2 y_s2  =      0.25 * s2 - im * 0.5 * s1 -   c * s0 + im * 0.5 * s_1 +     0.25 * s_2;
					double2 y_s1  = -im * 0.5 * s2 -      0.5 * s1            -      0.5 * s_1 + im * 0.5 * s_2;
					double2 y_s0  =        -c * s2                 - 0.5 * s0                  -        c * s_2;
					double2 y_s_1 =  im * 0.5 * s2 -      0.5 * s1            -      0.5 * s_1 - im * 0.5 * s_2;
					double2 y_s_2 =      0.25 * s2 + im * 0.5 * s1 -   c * s0 - im * 0.5 * s_1 +     0.25 * s_2;

					s2 = y_s2;
					s1 = y_s1;
					s0 = y_s0;
					s_1 = y_s_1;
					s_2 = y_s_2;
#endif

					double dens_s2 = s2.x * s2.x + s2.y * s2.y;
					double dens_s1 = s1.x * s1.x + s1.y * s1.y;
					double dens_s0 = s0.x * s0.x + s0.y * s0.y;
					double dens_s_1 = s_1.x * s_1.x + s_1.y * s_1.y;
					double dens_s_2 = s_2.x * s_2.x + s_2.y * s_2.y;
					double dens = dens_s2 + dens_s1 + dens_s0 + dens_s_1 + dens_s_2;

					float s0_r = 0;
					if (DENSITY_THRESHOLD < dens)
						s0_r = (float)(s0.x / sqrt(dens));
					swapEnd(s0_r);
					file.write((char*)&s0_r, sizeof(float));
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
					double2 s2 = pPsi[idx].values[dualNode].s2;
					double2 s1 = pPsi[idx].values[dualNode].s1;
					double2 s0 = pPsi[idx].values[dualNode].s0;
					double2 s_1 = pPsi[idx].values[dualNode].s_1;
					double2 s_2 = pPsi[idx].values[dualNode].s_2;
#if BASIS == X_QUANTIZED
					double c = sqrt(6) * 0.25;
					double2 x_s2  = 0.25 * s2 + 0.5 * s1 +   c * s0 + 0.5 * s_1 + 0.25 * s_2;
					double2 x_s1  = -0.5 * s2 - 0.5 * s1            + 0.5 * s_1 +  0.5 * s_2;
					double2 x_s0  =    c * s2            - 0.5 * s0             +    c * s_2;
					double2 x_s_1 = -0.5 * s2 + 0.5 * s1            - 0.5 * s_1 +  0.5 * s_2;
					double2 x_s_2 = 0.25 * s2 - 0.5 * s1 +   c * s0 - 0.5 * s_1 + 0.25 * s_2;

					s2 = x_s2;
					s1 = x_s1;
					s0 = x_s0;
					s_1 = x_s_1;
					s_2 = x_s_2;
#elif BASIS == Y_QUANTIZED
					double c = sqrt(6) * 0.25;
					double2 im = { 0, 1 };
					double2 y_s2  =      0.25 * s2 - im * 0.5 * s1 -   c * s0 + im * 0.5 * s_1 +     0.25 * s_2;
					double2 y_s1  = -im * 0.5 * s2 -      0.5 * s1            -      0.5 * s_1 + im * 0.5 * s_2;
					double2 y_s0  =        -c * s2                 - 0.5 * s0                  -        c * s_2;
					double2 y_s_1 =  im * 0.5 * s2 -      0.5 * s1            -      0.5 * s_1 - im * 0.5 * s_2;
					double2 y_s_2 =      0.25 * s2 + im * 0.5 * s1 -   c * s0 - im * 0.5 * s_1 +     0.25 * s_2;

					s2 = y_s2;
					s1 = y_s1;
					s0 = y_s0;
					s_1 = y_s_1;
					s_2 = y_s_2;
#endif

					double dens_s2 = s2.x * s2.x + s2.y * s2.y;
					double dens_s1 = s1.x * s1.x + s1.y * s1.y;
					double dens_s0 = s0.x * s0.x + s0.y * s0.y;
					double dens_s_1 = s_1.x * s_1.x + s_1.y * s_1.y;
					double dens_s_2 = s_2.x * s_2.x + s_2.y * s_2.y;
					double dens = dens_s2 + dens_s1 + dens_s0 + dens_s_1 + dens_s_2;

					float s0_i = 0;
					if (DENSITY_THRESHOLD < dens)
						s0_i = (float)(s0.y / sqrt(dens));
					swapEnd(s0_i);
					file.write((char*)&s0_i, sizeof(float));
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
					double2 s2 = pPsi[idx].values[dualNode].s2;
					double2 s1 = pPsi[idx].values[dualNode].s1;
					double2 s0 = pPsi[idx].values[dualNode].s0;
					double2 s_1 = pPsi[idx].values[dualNode].s_1;
					double2 s_2 = pPsi[idx].values[dualNode].s_2;
#if BASIS == X_QUANTIZED
					double c = sqrt(6) * 0.25;
					double2 x_s2  = 0.25 * s2 + 0.5 * s1 +   c * s0 + 0.5 * s_1 + 0.25 * s_2;
					double2 x_s1  = -0.5 * s2 - 0.5 * s1            + 0.5 * s_1 +  0.5 * s_2;
					double2 x_s0  =    c * s2            - 0.5 * s0             +    c * s_2;
					double2 x_s_1 = -0.5 * s2 + 0.5 * s1            - 0.5 * s_1 +  0.5 * s_2;
					double2 x_s_2 = 0.25 * s2 - 0.5 * s1 +   c * s0 - 0.5 * s_1 + 0.25 * s_2;

					s2 = x_s2;
					s1 = x_s1;
					s0 = x_s0;
					s_1 = x_s_1;
					s_2 = x_s_2;
#elif BASIS == Y_QUANTIZED
					double c = sqrt(6) * 0.25;
					double2 im = { 0, 1 };
					double2 y_s2  =      0.25 * s2 - im * 0.5 * s1 -   c * s0 + im * 0.5 * s_1 +     0.25 * s_2;
					double2 y_s1  = -im * 0.5 * s2 -      0.5 * s1            -      0.5 * s_1 + im * 0.5 * s_2;
					double2 y_s0  =        -c * s2                 - 0.5 * s0                  -        c * s_2;
					double2 y_s_1 =  im * 0.5 * s2 -      0.5 * s1            -      0.5 * s_1 - im * 0.5 * s_2;
					double2 y_s_2 =      0.25 * s2 + im * 0.5 * s1 -   c * s0 - im * 0.5 * s_1 +     0.25 * s_2;

					s2 = y_s2;
					s1 = y_s1;
					s0 = y_s0;
					s_1 = y_s_1;
					s_2 = y_s_2;
#endif

					double dens_s2 = s2.x * s2.x + s2.y * s2.y;
					double dens_s1 = s1.x * s1.x + s1.y * s1.y;
					double dens_s0 = s0.x * s0.x + s0.y * s0.y;
					double dens_s_1 = s_1.x * s_1.x + s_1.y * s_1.y;
					double dens_s_2 = s_2.x * s_2.x + s_2.y * s_2.y;
					double dens = dens_s2 + dens_s1 + dens_s0 + dens_s_1 + dens_s_2;

					float s_1_r = 0;
					if (DENSITY_THRESHOLD < dens)
						s_1_r = (float)(s_1.x / sqrt(dens));
					swapEnd(s_1_r);
					file.write((char*)&s_1_r, sizeof(float));
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
					double2 s2 = pPsi[idx].values[dualNode].s2;
					double2 s1 = pPsi[idx].values[dualNode].s1;
					double2 s0 = pPsi[idx].values[dualNode].s0;
					double2 s_1 = pPsi[idx].values[dualNode].s_1;
					double2 s_2 = pPsi[idx].values[dualNode].s_2;
#if BASIS == X_QUANTIZED
					double c = sqrt(6) * 0.25;
					double2 x_s2  = 0.25 * s2 + 0.5 * s1 +   c * s0 + 0.5 * s_1 + 0.25 * s_2;
					double2 x_s1  = -0.5 * s2 - 0.5 * s1            + 0.5 * s_1 +  0.5 * s_2;
					double2 x_s0  =    c * s2            - 0.5 * s0             +    c * s_2;
					double2 x_s_1 = -0.5 * s2 + 0.5 * s1            - 0.5 * s_1 +  0.5 * s_2;
					double2 x_s_2 = 0.25 * s2 - 0.5 * s1 +   c * s0 - 0.5 * s_1 + 0.25 * s_2;

					s2 = x_s2;
					s1 = x_s1;
					s0 = x_s0;
					s_1 = x_s_1;
					s_2 = x_s_2;
#elif BASIS == Y_QUANTIZED
					double c = sqrt(6) * 0.25;
					double2 im = { 0, 1 };
					double2 y_s2  =      0.25 * s2 - im * 0.5 * s1 -   c * s0 + im * 0.5 * s_1 +     0.25 * s_2;
					double2 y_s1  = -im * 0.5 * s2 -      0.5 * s1            -      0.5 * s_1 + im * 0.5 * s_2;
					double2 y_s0  =        -c * s2                 - 0.5 * s0                  -        c * s_2;
					double2 y_s_1 =  im * 0.5 * s2 -      0.5 * s1            -      0.5 * s_1 - im * 0.5 * s_2;
					double2 y_s_2 =      0.25 * s2 + im * 0.5 * s1 -   c * s0 - im * 0.5 * s_1 +     0.25 * s_2;

					s2 = y_s2;
					s1 = y_s1;
					s0 = y_s0;
					s_1 = y_s_1;
					s_2 = y_s_2;
#endif

					double dens_s2 = s2.x * s2.x + s2.y * s2.y;
					double dens_s1 = s1.x * s1.x + s1.y * s1.y;
					double dens_s0 = s0.x * s0.x + s0.y * s0.y;
					double dens_s_1 = s_1.x * s_1.x + s_1.y * s_1.y;
					double dens_s_2 = s_2.x * s_2.x + s_2.y * s_2.y;
					double dens = dens_s2 + dens_s1 + dens_s0 + dens_s_1 + dens_s_2;

					float s_1_i = 0;
					if (DENSITY_THRESHOLD < dens)
						s_1_i = (float)(s_1.y / sqrt(dens));
					swapEnd(s_1_i);
					file.write((char*)&s_1_i, sizeof(float));
				}
			}
		}
	}

	file << std::endl << "SCALARS r_m=-2 float 1" << std::endl;
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
					double2 s2 = pPsi[idx].values[dualNode].s2;
					double2 s1 = pPsi[idx].values[dualNode].s1;
					double2 s0 = pPsi[idx].values[dualNode].s0;
					double2 s_1 = pPsi[idx].values[dualNode].s_1;
					double2 s_2 = pPsi[idx].values[dualNode].s_2;
#if BASIS == X_QUANTIZED
					double c = sqrt(6) * 0.25;
					double2 x_s2  = 0.25 * s2 + 0.5 * s1 +   c * s0 + 0.5 * s_1 + 0.25 * s_2;
					double2 x_s1  = -0.5 * s2 - 0.5 * s1            + 0.5 * s_1 +  0.5 * s_2;
					double2 x_s0  =    c * s2            - 0.5 * s0             +    c * s_2;
					double2 x_s_1 = -0.5 * s2 + 0.5 * s1            - 0.5 * s_1 +  0.5 * s_2;
					double2 x_s_2 = 0.25 * s2 - 0.5 * s1 +   c * s0 - 0.5 * s_1 + 0.25 * s_2;

					s2 = x_s2;
					s1 = x_s1;
					s0 = x_s0;
					s_1 = x_s_1;
					s_2 = x_s_2;
#elif BASIS == Y_QUANTIZED
					double c = sqrt(6) * 0.25;
					double2 im = { 0, 1 };
					double2 y_s2  =      0.25 * s2 - im * 0.5 * s1 -   c * s0 + im * 0.5 * s_1 +     0.25 * s_2;
					double2 y_s1  = -im * 0.5 * s2 -      0.5 * s1            -      0.5 * s_1 + im * 0.5 * s_2;
					double2 y_s0  =        -c * s2                 - 0.5 * s0                  -        c * s_2;
					double2 y_s_1 =  im * 0.5 * s2 -      0.5 * s1            -      0.5 * s_1 - im * 0.5 * s_2;
					double2 y_s_2 =      0.25 * s2 + im * 0.5 * s1 -   c * s0 - im * 0.5 * s_1 +     0.25 * s_2;

					s2 = y_s2;
					s1 = y_s1;
					s0 = y_s0;
					s_1 = y_s_1;
					s_2 = y_s_2;
#endif

					double dens_s2 = s2.x * s2.x + s2.y * s2.y;
					double dens_s1 = s1.x * s1.x + s1.y * s1.y;
					double dens_s0 = s0.x * s0.x + s0.y * s0.y;
					double dens_s_1 = s_1.x * s_1.x + s_1.y * s_1.y;
					double dens_s_2 = s_2.x * s_2.x + s_2.y * s_2.y;
					double dens = dens_s2 + dens_s1 + dens_s0 + dens_s_1 + dens_s_2;

					float s_2_r = 0;
					if (DENSITY_THRESHOLD < dens)
						s_2_r = (float)(s_2.x / sqrt(dens));
					swapEnd(s_2_r);
					file.write((char*)&s_2_r, sizeof(float));
				}
			}
		}
	}
	file << std::endl << "SCALARS i_m=-2 float 1" << std::endl;
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
					double2 s2 = pPsi[idx].values[dualNode].s2;
					double2 s1 = pPsi[idx].values[dualNode].s1;
					double2 s0 = pPsi[idx].values[dualNode].s0;
					double2 s_1 = pPsi[idx].values[dualNode].s_1;
					double2 s_2 = pPsi[idx].values[dualNode].s_2;
#if BASIS == X_QUANTIZED
					double c = sqrt(6) * 0.25;
					double2 x_s2  = 0.25 * s2 + 0.5 * s1 +   c * s0 + 0.5 * s_1 + 0.25 * s_2;
					double2 x_s1  = -0.5 * s2 - 0.5 * s1            + 0.5 * s_1 +  0.5 * s_2;
					double2 x_s0  =    c * s2            - 0.5 * s0             +    c * s_2;
					double2 x_s_1 = -0.5 * s2 + 0.5 * s1            - 0.5 * s_1 +  0.5 * s_2;
					double2 x_s_2 = 0.25 * s2 - 0.5 * s1 +   c * s0 - 0.5 * s_1 + 0.25 * s_2;

					s2 = x_s2;
					s1 = x_s1;
					s0 = x_s0;
					s_1 = x_s_1;
					s_2 = x_s_2;
#elif BASIS == Y_QUANTIZED
					double c = sqrt(6) * 0.25;
					double2 im = { 0, 1 };
					double2 y_s2  =      0.25 * s2 - im * 0.5 * s1 -   c * s0 + im * 0.5 * s_1 +     0.25 * s_2;
					double2 y_s1  = -im * 0.5 * s2 -      0.5 * s1            -      0.5 * s_1 + im * 0.5 * s_2;
					double2 y_s0  =        -c * s2                 - 0.5 * s0                  -        c * s_2;
					double2 y_s_1 =  im * 0.5 * s2 -      0.5 * s1            -      0.5 * s_1 - im * 0.5 * s_2;
					double2 y_s_2 =      0.25 * s2 + im * 0.5 * s1 -   c * s0 - im * 0.5 * s_1 +     0.25 * s_2;

					s2 = y_s2;
					s1 = y_s1;
					s0 = y_s0;
					s_1 = y_s_1;
					s_2 = y_s_2;
#endif

					double dens_s2 = s2.x * s2.x + s2.y * s2.y;
					double dens_s1 = s1.x * s1.x + s1.y * s1.y;
					double dens_s0 = s0.x * s0.x + s0.y * s0.y;
					double dens_s_1 = s_1.x * s_1.x + s_1.y * s_1.y;
					double dens_s_2 = s_2.x * s_2.x + s_2.y * s_2.y;
					double dens = dens_s2 + dens_s1 + dens_s0 + dens_s_1 + dens_s_2;

					float s_2_i = 0;
					if (DENSITY_THRESHOLD < dens)
						s_2_i = (float)(s_2.y / sqrt(dens));
					swapEnd(s_2_i);
					file.write((char*)&s_2_i, sizeof(float));
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

	return com 	/ totDens;
}

#endif // UTILS