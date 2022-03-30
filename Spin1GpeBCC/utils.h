#ifndef UTILS
#define UTILS

#include <cuda_runtime.h>
#include <helper_cuda.h>

#include "VortexState.hpp"
#include <Output/Picture.hpp>
#include <Output/Text.hpp>
#include <Types/Complex.hpp>

#include <mesh.h>

// Arithmetic operators for cuda vector types
inline __host__ __device__ __inline__ double2 operator+(double2 a, double2 b)
{
	return { a.x + b.x, a.y + b.y };
}
inline __host__ __device__ __inline__ double3 operator+(double3 a, double3 b)
{
	return { a.x + b.x, a.y + b.y, a.z + b.z };
}
inline __host__ __device__ __inline__ double2 operator-(double2 a, double2 b)
{
	return { a.x - b.x, a.y - b.y };
}
inline __host__ __device__ __inline__ void operator+=(double2& a, double2 b)
{
	a.x += b.x;
	a.y += b.y;
}
inline __host__ __device__ __inline__ void operator+=(double3& a, double3 b)
{
	a.x += b.x;
	a.y += b.y;
	a.z += b.z;
}
inline __host__ __device__ __inline__ void operator-=(double2& a, double2 b)
{
	a.x -= b.x;
	a.y -= b.y;
}
inline __host__ __device__ __inline__ double2 operator*(double b, double2 a)
{
	return { b * a.x, b * a.y };
}
inline __host__ __device__ __inline__ double3 operator*(double b, double3 a)
{
	return { b * a.x, b * a.y, b * a.z };
}
inline __host__ __device__ __inline__ double3 operator/(double3 a, double b)
{
	return { a.x / b, a.y / b, a.z / b };
}
inline __host__ __device__ __inline__ double2 operator/(double2 a, double b)
{
	return { a.x / b, a.y / b };
}
inline __host__ __device__ __inline__ double2 conj(double2 a) // Complex conjugate
{
	return { a.x, -a.y };
}
inline __host__ __device__ __inline__ double2 operator*(double2 a, double2 b) // Complex number multiplication
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

struct PitchedPtr
{
	char* __restrict__ ptr;
	size_t pitch;
	size_t slicePitch;
};

struct MagFields
{
	double Bq{};
	double Bz{};
	double BqQuad{};
	double BzQuad{};
};

std::string toString(const double value)
{
	std::ostringstream out;
	out.precision(18);
	out << std::fixed << value;
	return out.str();
};

void drawDensity(const std::string& name, BlockPsis* h_evenPsi, size_t dxsize, size_t dysize, size_t dzsize, double t)
{
	const int SIZE = 2;
	const double INTENSITY = 1.0;

	const int width = dxsize * SIZE, height = dysize * SIZE, depth = dzsize * SIZE;
	Picture pic1(width * 3, height * 2);

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

			pic1.setColor(i, k, Vector4(s1, s1, s1, 1.0));
			pic1.setColor(width + i, k, Vector4(s0, s0, s0, 1.0));
			pic1.setColor(2 * width + i, k, Vector4(s_1, s_1, s_1, 1.0));
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

			pic1.setColor(i, height + j, Vector4(s1, s1, s1, 1.0));
			pic1.setColor(width + i, height + j, Vector4(s0, s0, s0, 1.0));
			pic1.setColor(2 * width + i, height + j, Vector4(s_1, s_1, s_1, 1.0));
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

	pic1.save("results/" + name + toString(t) + "ms.bmp", false);
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

bool saveVolumeMap(const std::string& path, const Buffer<ushort>& vol, const uint xsize, const uint ysize, const uint zsize, const Vector3& h)
{
	Text rawpath;
	rawpath << path << ".raw";

	// save raw
	std::ofstream fs(rawpath.str().c_str(), std::ios_base::binary | std::ios::trunc);
	if (fs.fail()) return false;
	fs.write((char*)&vol[0], 2 * xsize * ysize * zsize);
	fs.close();

	// save header
	Text text;

	text << "ObjectType              = Image" << std::endl;
	text << "NDims                   = 3" << std::endl;
	text << "BinaryData              = True" << std::endl;
	text << "CompressedData          = False" << std::endl;
	text << "BinaryDataByteOrderMSB  = False" << std::endl;
	text << "TransformMatrix         = 1 0 0 0 1 0 0 0 1" << std::endl;
	text << "Offset                  = " << -0.5 * xsize * h.x << " " << -0.5 * ysize * h.y << " " << -0.5 * zsize * h.z << std::endl;
	text << "CenterOfRotation        = 0 0 0" << std::endl;
	text << "DimSize                 = " << xsize << " " << ysize << " " << zsize << std::endl;
	text << "ElementSpacing          = " << h.x << " " << h.y << " " << h.z << std::endl;
	text << "ElementNumberOfChannels = 1" << std::endl;
	text << "ElementType             = MET_USHORT" << std::endl;
	text << "ElementDataFile         = " << rawpath.str() << std::endl;
	text.save(path);
	return true;
}

void saveVolume(const std::string& name, BlockPsis* h_evenPsi, size_t bsize, size_t dxsize, size_t dysize, size_t dzsize, uint iter, double block_scale)
{
	// save volume map
	const double fmax = 1.0f; // state.searchFunctionMax();
	const double unit = 60000.0 / (bsize * fmax * fmax);
	Buffer<ushort> vol(dxsize * dysize * dzsize);
	for (uint k = 0; k < dzsize; k++)
	{
		for (uint j = 0; j < dysize; j++)
		{
			for (uint i = 0; i < dxsize; i++)
			{
				const uint idx = k * dxsize * dysize + j * dxsize + i;
				double sum = 0.0;
				for (uint l = 0; l < bsize; l++)
				{
					sum += h_evenPsi[idx].values[0].s1.x * h_evenPsi[idx].values[0].s1.x + h_evenPsi[idx].values[0].s1.y * h_evenPsi[idx].values[0].s1.y;
				}
				sum *= unit;
				vol[idx] = (sum > 65535.0 ? 65535 : ushort(sum));
			}
		}
	}
	Text volpath;
	volpath << "volume" << iter << ".mhd";
	saveVolumeMap(volpath.str(), vol, dxsize, dysize, dzsize, block_scale * BLOCK_WIDTH);
}

#endif // UTILS