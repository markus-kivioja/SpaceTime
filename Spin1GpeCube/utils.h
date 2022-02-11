#ifndef UTILS
#define UTILS

#include <cuda_runtime.h>
#include <helper_cuda.h>

#include "VortexState.hpp"
#include <Output/Picture.hpp>
#include <Output/Text.hpp>
#include <Types/Complex.hpp>

#include <mesh.h>

struct Complex3Vec
{
	double2 s1 = make_double2(0, 0);
	double2 s0 = make_double2(0, 0);
	double2 s_1 = make_double2(0, 0);
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

void drawPicture(const std::string& name, BlockPsis* h_evenPsi, size_t dxsize, size_t dysize, size_t dzsize, uint iter, double Bq, double Bz, double block_scale, double3 p0)
{
	const int SIZE = 2;

	double intensity_s1 = 0;
	double intensity_s0 = 0;
	double intensity_s_1 = 0;
	int width = dxsize * SIZE, height = dysize * SIZE, depth = dzsize * SIZE;
	Picture pic1(width * 2, height);
	{
		//Picture pic0(width, height);
		//Picture pic_1(width, height);
		//uint k = dzsize / 2 + 1;
		//for (uint j = 0; j < height; j++)
		//{
		//	for (uint i = 0; i < width; i++)
		//	{
		//		const uint idx = k * dxsize * dysize + (j / SIZE) * dxsize + i / SIZE;
		//		double norm_s1 = h_evenPsi[idx].values[0].s1.x * h_evenPsi[idx].values[0].s1.x + h_evenPsi[idx].values[0].s1.y * h_evenPsi[idx].values[0].s1.y;
		//		double norm_s0 = h_evenPsi[idx].values[0].s0.x * h_evenPsi[idx].values[0].s0.x + h_evenPsi[idx].values[0].s0.y * h_evenPsi[idx].values[0].s0.y;
		//		double norm_s_1 = h_evenPsi[idx].values[0].s_1.x * h_evenPsi[idx].values[0].s_1.x + h_evenPsi[idx].values[0].s_1.y * h_evenPsi[idx].values[0].s_1.y;
		//
		//		intensity_s1 = max(intensity_s1, norm_s1);
		//		intensity_s0 = max(intensity_s0, norm_s0);
		//		intensity_s_1 = max(intensity_s_1, norm_s_1);
		//	}
		//}
		//std::cout << intensity_s1 << ", " << intensity_s0 << ", " << intensity_s_1 << std::endl;

		intensity_s1  = 5.0; // 1.0 / intensity_s1;
		intensity_s0  = 5.0; // 1.0 / intensity_s0;
		intensity_s_1 = 5.0; // 1.0 / intensity_s_1;
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
					norm_s1 += h_evenPsi[idx].values[0].s1.x * h_evenPsi[idx].values[0].s1.x + h_evenPsi[idx].values[0].s1.y * h_evenPsi[idx].values[0].s1.y;
					norm_s0 += h_evenPsi[idx].values[0].s0.x * h_evenPsi[idx].values[0].s0.x + h_evenPsi[idx].values[0].s0.y * h_evenPsi[idx].values[0].s0.y;
					norm_s_1 += h_evenPsi[idx].values[0].s_1.x * h_evenPsi[idx].values[0].s_1.x + h_evenPsi[idx].values[0].s_1.y * h_evenPsi[idx].values[0].s_1.y;
				}

				double r = intensity_s1 * norm_s1;
				double g = intensity_s_1 * norm_s_1;
				double b = intensity_s0 * norm_s0;

				pic1.setColor(i, j, Vector4(r, g, b, 1.0));
				//pic1.setColor(i, j, intensity_s1 * Vector4(norm_s1, 0, 0, 1.0));
				//pic0.setColor(i, j, intensity_s0 * Vector4(0, norm_s0, 0, 1.0));
				//pic_1.setColor(i, j, intensity_s_1 * Vector4(0, 0, norm_s_1, 1.0));
			}
		}
		intensity_s1 = 3.0; // 1.0 / intensity_s1;
		intensity_s0 = 3.0; // 1.0 / intensity_s0;
		intensity_s_1 = 3.0; // 1.0 / intensity_s_1;
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
					norm_s1 += h_evenPsi[idx].values[0].s1.x * h_evenPsi[idx].values[0].s1.x + h_evenPsi[idx].values[0].s1.y * h_evenPsi[idx].values[0].s1.y;
					norm_s0 += h_evenPsi[idx].values[0].s0.x * h_evenPsi[idx].values[0].s0.x + h_evenPsi[idx].values[0].s0.y * h_evenPsi[idx].values[0].s0.y;
					norm_s_1 += h_evenPsi[idx].values[0].s_1.x * h_evenPsi[idx].values[0].s_1.x + h_evenPsi[idx].values[0].s_1.y * h_evenPsi[idx].values[0].s_1.y;
				}

				double r = intensity_s1 * norm_s1;
				double g = intensity_s_1 * norm_s_1;
				double b = intensity_s0 * norm_s0;

				double3 localPos = getLocalPos(0);
				double3 globalPos = make_double3(p0.x + block_scale * ((double)(i / SIZE) * H_BLOCK_WIDTH_X + localPos.x),
					0,
					p0.z + block_scale * ((k / SIZE) * H_BLOCK_WIDTH_Z + localPos.z));
				double3 B = magneticField(globalPos, Bq, Bz);
				double normB = sqrt(B.x * B.x + B.y * B.y + B.z * B.z);
				if (normB < 0.1)
					pic1.setColor(width + k, i, Vector4(1, 1, 1, 1.0));
				else
					pic1.setColor(width + k, i, Vector4(r, g, b, 1.0));
				//pic1.setColor(k, i, intensity_s1 * Vector4(norm_s1, 0, 0, 1.0));
				//pic0.setColor(k, i, intensity_s0 * Vector4(0, norm_s0, 0, 1.0));
				//pic_1.setColor(k, i, intensity_s_1 * Vector4(0, 0, norm_s_1, 1.0));
			}
		}
		pic1.save("results/" + name + "_" + std::to_string(iter) + "_s1.bmp", false);
		//pic0.save("results/" + name + "_" + std::to_string(iter) + "_s0.bmp", false);
		//pic_1.save("results/" + name + "_" + std::to_string(iter) + "_s-1.bmp", false);
	}
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

// Arithmetic operators for cuda vector types
inline __host__ __device__ __inline__ double2 operator+(double2 a, double2 b)
{
	return make_double2(a.x + b.x, a.y + b.y);
}
inline __host__ __device__ __inline__ double2 operator-(double2 a, double2 b)
{
	return make_double2(a.x - b.x, a.y - b.y);
}
inline __host__ __device__ __inline__ void operator+=(double2& a, double2 b)
{
	a.x += b.x;
	a.y += b.y;
}
inline __host__ __device__ __inline__ void operator-=(double2& a, double2 b)
{
	a.x -= b.x;
	a.y -= b.y;
}
inline __host__ __device__ __inline__ double2 operator*(double b, double2 a)
{
	return make_double2(b * a.x, b * a.y);
}
inline __host__ __device__ __inline__ double2 operator/(double2 a, double b)
{
	return make_double2(a.x / b, a.y / b);
}
inline __host__ __device__ __inline__ double2 star(double2 a) // Complex conjugate
{
	return make_double2(a.x, -a.y);
}
inline __host__ __device__ __inline__ double2 operator*(double2 a, double2 b) // Complex number multiplication
{
	return make_double2(a.x * b.x - a.y * b.y, a.y * b.x + a.x * b.y);
}

#endif // UTILS