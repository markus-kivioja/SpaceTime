#include <cuda_runtime.h>
#include <helper_cuda.h>

#include "VortexState.hpp"
#include <Output/Picture.hpp>
#include <Output/Text.hpp>
#include <Types/Complex.hpp>
#include <Types/Random.hpp>
#include <Mesh/DelaunayMesh.hpp>
#include <iostream>
#include <sstream>

#include <mesh.h>

static constexpr ddouble RATIO = 1.0;
static constexpr ddouble C0 = 7500.0;
static constexpr ddouble C1 = -75.0;
static constexpr ddouble gF = -0.5; // Lande g factor

// The external magnetic field
static constexpr ddouble Bq = -0.05; // Quadrupole gradient TODO: Scale from T/m to diemensionless
static constexpr ddouble Bz0 = 55.985f; // Bias field at t = 0 TODO: Scale from uT to dimensionless
static constexpr ddouble BzVelocity = 0; // Time derivative of the bias field TODO: Set

#define INV_SQRT_2 0.70710678118655

#define LOAD_STATE_FROM_DISK 0
#define SAVE_PICTURE 1
#define SAVE_VOLUME 0

#define THREAD_BLOCK_X 8
#define THREAD_BLOCK_Y 8
#define THREAD_BLOCK_Z 1

ddouble potentialRZ(const ddouble r, const ddouble z)
{
	return 0.5 * (r * r + RATIO * RATIO * z * z);
}

ddouble potentialV3(const Vector3& p)
{
	return 0.5 * (p.x * p.x + p.y * p.y + RATIO * RATIO * p.z * p.z);
}

Vector3 magneticField(const Vector3& p, ddouble t)
{
	ddouble Bz = Bz0 + BzVelocity * t;

	return Bq * Vector3(p.x, p.y, -2 * p.z + Bz);
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

struct BlockPots
{
	double values[VALUES_IN_BLOCK];
};

struct BlockBs
{
	double3 values[VALUES_IN_BLOCK];
};

struct PitchedPtr
{
	char* ptr;
	size_t pitch;
	size_t slicePitch;
};

// Arithmetic operators for cuda vector types
inline __host__ __device__ double2 operator+(double2 a, double2 b)
{
	return make_double2(a.x + b.x, a.y + b.y);
}
inline __host__ __device__ double2 operator-(double2 a, double2 b)
{
	return make_double2(a.x - b.x, a.y - b.y);
}
inline __host__ __device__ void operator+=(double2& a, double2 b)
{
	a.x += b.x;
	a.y += b.y;
}
inline __host__ __device__ double2 operator*(double b, double2 a)
{
	return make_double2(b * a.x, b * a.y);
}
inline __host__ __device__ double2 star(double2 a) // Complex conjugate
{
	return make_double2(a.x, -a.y);
}
inline __host__ __device__ double2 operator*(double2 a, double2 b) // Complex number multiplication
{
	return make_double2(a.x * b.x - a.y * b.y, a.y * b.x + a.x * b.y);
}

__global__ void update(PitchedPtr nextStep, PitchedPtr prevStep, PitchedPtr trapPots, PitchedPtr Bptr, int2* lapInd, double* hodges, ddouble c0, ddouble c1, uint3 dimensions)
{
	size_t xid = blockIdx.x * blockDim.x + threadIdx.x;
	size_t yid = blockIdx.y * blockDim.y + threadIdx.y;
	size_t zid = blockIdx.z * blockDim.z + threadIdx.z;

	size_t dataZid = zid / VALUES_IN_BLOCK; // One thread per every dual node so VALUES_IN_BLOCK threads per mesh block (on z-axis)

	// Exit leftover threads
	if (xid >= dimensions.x || yid >= dimensions.y || dataZid >= dimensions.z)
	{
		return;
	}

	// Calculate the pointers for this block
	char* prevPsi = prevStep.ptr + prevStep.slicePitch * dataZid + prevStep.pitch * yid + sizeof(BlockPsis) * xid;
	BlockPsis* nextPsi = (BlockPsis*)(nextStep.ptr + nextStep.slicePitch * dataZid + nextStep.pitch * yid) + xid;
	BlockPots* trapPot = (BlockPots*)(trapPots.ptr + trapPots.slicePitch * dataZid + trapPots.pitch * yid) + xid;
	BlockBs* magField = (BlockBs*)(Bptr.ptr + Bptr.slicePitch * dataZid + Bptr.pitch * yid) + xid;

	// Update psi
	size_t dualNodeId = zid % VALUES_IN_BLOCK; // Dual node id. One thread per every dual node so VALUES_IN_BLOCK threads per mesh block (on z-axis)
	Complex3Vec prev = ((BlockPsis*)prevPsi)->values[dualNodeId];

	uint primaryFace = dualNodeId * FACE_COUNT;
	
	Complex3Vec H;
	H.s1 = make_double2(0, 0);
	H.s0 = make_double2(0, 0);
	H.s_1 = make_double2(0, 0);

	// Add Laplacian to Hamiltonian
#pragma unroll
	for (int i = 0; i < FACE_COUNT; ++i)
	{
		H.s1 += hodges[primaryFace] * (((BlockPsis*)(prevPsi + lapInd[primaryFace].x))->values[lapInd[primaryFace++].y].s1 - prev.s1);
		H.s0 += hodges[primaryFace] * (((BlockPsis*)(prevPsi + lapInd[primaryFace].x))->values[lapInd[primaryFace++].y].s0 - prev.s0);
		H.s_1 += hodges[primaryFace] * (((BlockPsis*)(prevPsi + lapInd[primaryFace].x))->values[lapInd[primaryFace++].y].s_1 - prev.s_1);
	}

	double normSq_s1 = prev.s1.x * prev.s1.x + prev.s1.y * prev.s1.y;
	double normSq_s0 = prev.s0.x * prev.s0.x + prev.s0.y * prev.s0.y;
	double normSq_s_1 = prev.s_1.x * prev.s_1.x + prev.s_1.y * prev.s_1.y;
	double normSq = normSq_s1 + normSq_s0 + normSq_s_1;

	// Add the total potential to Hamiltonian
	double totalPot = trapPot->values[dualNodeId] + (c0 + c1) * normSq;
	H.s1 += totalPot * prev.s1 + c1 * (-2 * normSq_s_1 * prev.s1 + star(prev.s_1) * prev.s0 * prev.s0 + 0 * prev.s_1);
	H.s0 += totalPot * prev.s0 + c1 * (star(prev.s0) * prev.s_1 * prev.s1 - normSq_s0 * prev.s0 + star(prev.s0) * prev.s1 * prev.s_1);
	H.s_1 += totalPot * prev.s_1 + c1 * (0 * prev.s1 + star(prev.s1) * prev.s0 * prev.s0 - 2 * normSq_s1 * prev.s_1);

	// Add the Zeeman term
	double3 B = magField->values[dualNodeId];
	H.s1 += gF * (B.z * prev.s1 + INV_SQRT_2 * make_double2(B.x, -B.y) * prev.s0);
	H.s0 += gF * (INV_SQRT_2 * make_double2(B.x, B.y) * prev.s1 + INV_SQRT_2 * make_double2(B.x, -B.y) * prev.s_1);
	H.s_1 += gF * (INV_SQRT_2 * make_double2(B.x, B.y) * prev.s0 - B.z * prev.s_1);

	nextPsi->values[dualNodeId].s1 += make_double2(H.s1.y, -H.s1.x);
	nextPsi->values[dualNodeId].s0 += make_double2(H.s0.y, -H.s0.x);
	nextPsi->values[dualNodeId].s_1 += make_double2(H.s_1.y, -H.s_1.x);
};

uint integrateInTime(const VortexState& state, const ddouble block_scale, const Vector3& minp, const Vector3& maxp, const ddouble iteration_period, const uint number_of_iterations)
{
	uint i, j, k, l;

	// find dimensions
	const Vector3 domain = maxp - minp;
	const uint xsize = uint(domain.x / (block_scale * BLOCK_WIDTH.x)) + 1;
	const uint ysize = uint(domain.y / (block_scale * BLOCK_WIDTH.y)) + 1;
	const uint zsize = uint(domain.z / (block_scale * BLOCK_WIDTH.z)) + 1;
	const Vector3 p0 = 0.5 * (minp + maxp - block_scale * Vector3(BLOCK_WIDTH.x * xsize, BLOCK_WIDTH.y * ysize, BLOCK_WIDTH.z * zsize));

	//std::cout << xsize << ", " << ysize << ", " << zsize << std::endl;

	// find relative circumcenters for each body element
	Buffer<Vector3> bpos;
	getPositions(bpos);

	// compute discrete dimensions
	const uint bsize = bpos.size(); // number of values inside a block
	const uint bxsize = (xsize + 1) * bsize; // number of values on x-row
	const uint bxysize = (ysize + 1) * bxsize; // number of values on xy-plane
	const uint ii0 = (IS_3D ? bxysize : 0) + bxsize + bsize; // reserved zeros in the beginning of value table
	const uint vsize = ii0 + (IS_3D ? zsize + 1 : zsize) * bxysize; // total number of values

	std::cout << "bsize: " << bsize << ", xsize: " << xsize << ", yszie: " << ysize << ", zsize: " << zsize << std::endl;
	uint64_t bodies = xsize * ysize * zsize * bsize;
	std::cout << "bodies = " << bodies << std::endl;

	// initialize stationary state
	Buffer<Complex> Psi0(vsize, Complex(0, 0)); // initial discrete wave function
	Buffer<ddouble> pot(vsize, 0.0); // discrete potential multiplied by time step size
	ddouble c0 = state.getC0(); // effective interaction strength
	ddouble c1 = state.getC1(); // effective interaction strength
	ddouble maxpot = 0.0; // maximal value of potential
	for (k = 0; k < zsize; k++)
	{
		for (j = 0; j < ysize; j++)
		{
			for (i = 0; i < xsize; i++)
			{
				for (l = 0; l < bsize; l++)
				{
					const uint ii = ii0 + k * bxysize + j * bxsize + i * bsize + l;
					const Vector3 p(p0.x + block_scale * (i * BLOCK_WIDTH.x + bpos[l].x), p0.y + block_scale * (j * BLOCK_WIDTH.y + bpos[l].y), p0.z + block_scale * (k * BLOCK_WIDTH.z + bpos[l].z)); // position
					Psi0[ii] = state.getPsi(p);
					pot[ii] = potentialV3(p);
					const ddouble poti = pot[ii] + g * Psi0[ii].normsq();
					if (poti > maxpot) maxpot = poti;
				}
			}
		}
	}

	// Initialize device memory
	size_t dxsize = xsize + 2; // One element buffer to both ends
	size_t dysize = ysize + 2; // One element buffer to both ends
	size_t dzsize = zsize + 2; // One element buffer to both ends
	cudaExtent psiExtent = make_cudaExtent(dxsize * sizeof(BlockPsis), dysize, dzsize);
	cudaExtent potExtent = make_cudaExtent(dxsize * sizeof(BlockPots), dysize, dzsize);

	cudaPitchedPtr d_cudaEvenPsi;
	cudaPitchedPtr d_cudaOddPsi;
	cudaPitchedPtr d_cudaPot;

	checkCudaErrors(cudaMalloc3D(&d_cudaEvenPsi, psiExtent));
	checkCudaErrors(cudaMalloc3D(&d_cudaOddPsi, psiExtent));
	checkCudaErrors(cudaMalloc3D(&d_cudaPot, potExtent));

	size_t offset = d_cudaEvenPsi.pitch * dysize + d_cudaEvenPsi.pitch + sizeof(BlockPsis);
	size_t potOffset = d_cudaPot.pitch * dysize + d_cudaPot.pitch + sizeof(BlockPots);
	PitchedPtr d_evenPsi = { (char*)d_cudaEvenPsi.ptr + offset, d_cudaEvenPsi.pitch, d_cudaEvenPsi.pitch * dysize };
	PitchedPtr d_oddPsi = { (char*)d_cudaOddPsi.ptr + offset, d_cudaOddPsi.pitch, d_cudaOddPsi.pitch * dysize };
	PitchedPtr d_pot = { (char*)d_cudaPot.ptr + potOffset, d_cudaPot.pitch, d_cudaPot.pitch * dysize };

	// find terms for laplacian
	Buffer<int2> lapind;
	Buffer<ddouble> hodges;
	ddouble lapfac = -0.5 * getLaplacian(lapind, hodges, sizeof(BlockPsis), d_evenPsi.pitch, d_evenPsi.slicePitch) / (block_scale * block_scale);
	const uint lapsize = lapind.size() / bsize;
	ddouble lapfac0 = lapsize * (-lapfac);

	//std::cout << "lapsize = " << lapsize << ", lapfac = " << lapfac << ", lapfac0 = " << lapfac0 << std::endl;

	// compute time step size
	const uint steps_per_iteration = uint(iteration_period * (maxpot + lapfac0)) + 1; // number of time steps per iteration period
	const ddouble time_step_size = iteration_period / ddouble(steps_per_iteration); // time step in time units

	std::cout << "steps_per_iteration = " << steps_per_iteration << std::endl;
	std::cout << "ALU operations per unit time = " << bodies * steps_per_iteration * FACE_COUNT << std::endl;
	std::cout << "Domain: " << domain.x << ", " << domain.y << ", " << domain.z << std::endl;
	std::cout << "Maxpot:" << maxpot << std::endl;

	// multiply terms with time_step_size
	c0 *= time_step_size;
	c1 *= time_step_size;
	lapfac *= time_step_size;
	lapfac0 *= time_step_size;
	for (i = 0; i < vsize; i++) pot[i] *= time_step_size;
	for (int i = 0; i < hodges.size(); ++i) hodges[i] = -0.5 * hodges[i] / (block_scale * block_scale) * time_step_size;

	int2* d_lapind;
	checkCudaErrors(cudaMalloc(&d_lapind, lapind.size() * sizeof(int2)));

	ddouble* d_hodges;
	checkCudaErrors(cudaMalloc(&d_hodges, hodges.size() * sizeof(ddouble)));

	// Initialize host memory
	size_t hostSize = dxsize * dysize * (zsize + 2);
	BlockPsis* h_evenPsi;// = new BlockPsis[dxsize * dysize * (zsize + 2)];
	BlockPsis* h_oddPsi;// = new BlockPsis[dxsize * dysize * (zsize + 2)];
	BlockPots* h_pot;// = new BlockPots[dxsize * dysize * (zsize + 2)];
	checkCudaErrors(cudaMallocHost(&h_evenPsi, hostSize * sizeof(BlockPsis)));
	checkCudaErrors(cudaMallocHost(&h_oddPsi, hostSize * sizeof(BlockPsis)));
	checkCudaErrors(cudaMallocHost(&h_pot, hostSize * sizeof(BlockPots)));
	memset(h_evenPsi, 0, hostSize * sizeof(BlockPsis));
	memset(h_oddPsi, 0, hostSize * sizeof(BlockPsis));
	memset(h_pot, 0, hostSize * sizeof(BlockPots));

	// initialize discrete field
	const Complex oddPhase = state.getPhase(-0.5 * time_step_size);
	//Random rnd(54363);
	for (k = 0; k < zsize; k++)
	{
		for (j = 0; j < ysize; j++)
		{
			for (i = 0; i < xsize; i++)
			{
				for (l = 0; l < bsize; l++)
				{
					const uint srcI = ii0 + k * bxysize + j * bxsize + i * bsize + l;
					const uint dstI = (k + 1) * dxsize * dysize + (j + 1) * dxsize + (i + 1);
					//const Vector2 c = 0.01 * rnd.getUniformCircle();
					//const Complex noise(c.x + 1.0, c.y);
					//const Complex noisedPsi = Psi0[srcI] * noise;
					//double2 even = make_double2(noisedPsi.r, noisedPsi.i);
					double2 even = make_double2(Psi0[srcI].r, Psi0[srcI].i); // No noice
					h_evenPsi[dstI].values[l] = even;
					h_oddPsi[dstI].values[l] = make_double2(oddPhase.r * even.x - oddPhase.i * even.y,
						oddPhase.i * even.x + oddPhase.r * even.y);
					h_pot[dstI].values[l] = pot[srcI];
				}
			}
		}
	}

	cudaPitchedPtr h_cudaEvenPsi = { 0 };
	cudaPitchedPtr h_cudaOddPsi = { 0 };
	cudaPitchedPtr h_cudaPot = { 0 };

	h_cudaEvenPsi.ptr = h_evenPsi;
	h_cudaEvenPsi.pitch = dxsize * sizeof(BlockPsis);
	h_cudaEvenPsi.xsize = d_cudaEvenPsi.xsize;
	h_cudaEvenPsi.ysize = d_cudaEvenPsi.ysize;

	h_cudaOddPsi.ptr = h_oddPsi;
	h_cudaOddPsi.pitch = dxsize * sizeof(BlockPsis);
	h_cudaOddPsi.xsize = d_cudaOddPsi.xsize;
	h_cudaOddPsi.ysize = d_cudaOddPsi.ysize;

	h_cudaPot.ptr = h_pot;
	h_cudaPot.pitch = dxsize * sizeof(BlockPots);
	h_cudaPot.xsize = d_cudaPot.xsize;
	h_cudaPot.ysize = d_cudaPot.ysize;

	// Copy from host memory to device memory
	cudaMemcpy3DParms evenPsiParams = { 0 };
	cudaMemcpy3DParms oddPsiParams = { 0 };
	cudaMemcpy3DParms potParams = { 0 };

	evenPsiParams.srcPtr = h_cudaEvenPsi;
	evenPsiParams.dstPtr = d_cudaEvenPsi;
	evenPsiParams.extent = psiExtent;
	evenPsiParams.kind = cudaMemcpyHostToDevice;

	oddPsiParams.srcPtr = h_cudaOddPsi;
	oddPsiParams.dstPtr = d_cudaOddPsi;
	oddPsiParams.extent = psiExtent;
	oddPsiParams.kind = cudaMemcpyHostToDevice;

	potParams.srcPtr = h_cudaPot;
	potParams.dstPtr = d_cudaPot;
	potParams.extent = potExtent;
	potParams.kind = cudaMemcpyHostToDevice;

	checkCudaErrors(cudaMemcpy3D(&evenPsiParams));
	checkCudaErrors(cudaMemcpy3D(&oddPsiParams));
	checkCudaErrors(cudaMemcpy3D(&potParams));
	checkCudaErrors(cudaMemcpy(d_lapind, &lapind[0], lapind.size() * sizeof(int2), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_hodges, &hodges[0], hodges.size() * sizeof(ddouble), cudaMemcpyHostToDevice));

	// Clear host memory after data has been copied to devices
	cudaDeviceSynchronize();
	//Psi0.clear();
	pot.clear();
	bpos.clear();
	lapind.clear();
	hodges.clear();
	cudaFreeHost(h_oddPsi);
	cudaFreeHost(h_pot);
#if !(SAVE_PICTURE || SAVE_VOLUME)
	cudaFreeHost(h_evenPsi);
#endif

	// Integrate in time
	uint3 dimensions = make_uint3(xsize, ysize, zsize);
	uint iter = 0;
	dim3 dimBlock(THREAD_BLOCK_X, THREAD_BLOCK_Y, THREAD_BLOCK_Z);
	dim3 dimGrid((xsize + THREAD_BLOCK_X - 1) / THREAD_BLOCK_X,
		(ysize + THREAD_BLOCK_Y - 1) / THREAD_BLOCK_Y,
		((zsize + THREAD_BLOCK_Z - 1) / THREAD_BLOCK_Z) * VALUES_IN_BLOCK);
#if SAVE_PICTURE || SAVE_VOLUME
	cudaMemcpy3DParms evenPsiBackParams = { 0 };
	evenPsiBackParams.srcPtr = d_cudaEvenPsi;
	evenPsiBackParams.dstPtr = h_cudaEvenPsi;
	evenPsiBackParams.extent = psiExtent;
	evenPsiBackParams.kind = cudaMemcpyDeviceToHost;
#endif
	Text errorText;
	const uint time0 = clock();
	const ddouble volume = (IS_3D ? block_scale : 1.0) * block_scale * block_scale * VOLUME;
	while (true)
	{
#if SAVE_PICTURE
		// draw picture
		const float INTENSITY = 20.0f;
		const int SIZE = 2;
		int width = dxsize * SIZE, height = dysize * SIZE;
		Picture pic(width, height);
		k = zsize / 2 + 1;
		for (j = 0; j < height; j++)
		{
			for (i = 0; i < width; i++)
			{
				const uint idx = k * dxsize * dysize + (j / SIZE) * dxsize + i / SIZE;
				double norm = sqrt(h_evenPsi[idx].values[0].x * h_evenPsi[idx].values[0].x + h_evenPsi[idx].values[0].y * h_evenPsi[idx].values[0].y);
		
				pic.setColor(i, j, INTENSITY * Vector4(h_evenPsi[idx].values[0].x, norm, h_evenPsi[idx].values[0].y, 1.0));
			}
		}
		std::ostringstream picpath;
		picpath << "results/kuva" << iter << ".bmp";
		pic.save(picpath.str(), false);

		// print squared norm and error
		const Complex currentPhase = state.getPhase(iter * steps_per_iteration * time_step_size);
		ddouble errorNormSq = 0;
		ddouble normsq = 0.0;
		Complex error(0.0, 0.0);
		ddouble maxError = 0;
		for (k = 0; k < zsize; k++)
		{
			for (j = 0; j < ysize; j++)
			{
				for (i = 0; i < xsize; i++)
				{
					for (l = 0; l < bsize; l++)
					{
						const uint srcI = ii0 + k * bxysize + j * bxsize + i * bsize + l;
						const uint dstI = (k + 1) * dxsize * dysize + (j + 1) * dxsize + (i + 1);

						Complex evenPsi(h_evenPsi[dstI].values[l].x, h_evenPsi[dstI].values[l].y);
						normsq += evenPsi.normsq() * volume;
						error += (Psi0[srcI].con() * evenPsi) * volume;

						Complex groundTruth = currentPhase * Psi0[srcI];
						ddouble curError = (groundTruth - evenPsi).normsq();
						errorNormSq += curError;

						if (curError > maxError) maxError = curError;
					}
				}
			}
		}
		ddouble RMSE = sqrt(errorNormSq / (double)(zsize * ysize * xsize * bsize));
		ddouble errorAbs = abs(normsq - error.norm());
		std::cout << "normsq=" << normsq << " error=" << sqrt(errorNormSq) << " max error: " << sqrt(maxError) <<  std::endl;
		errorText << RMSE << " ";
#endif

#if SAVE_VOLUME
		// save volume map
		const ddouble fmax = state.searchFunctionMax();
		const ddouble unit = 60000.0 / (bsize * fmax * fmax);
		Buffer<ushort> vol(dxsize * dysize * dzsize);
		for (k = 0; k < dzsize; k++)
		{
			for (j = 0; j < dysize; j++)
			{
				for (i = 0; i < dxsize; i++)
				{
					const uint idx = k * dxsize * dysize + j * dxsize + i;
					ddouble sum = 0.0;
					for (l = 0; l < bsize; l++)
					{
						sum += h_evenPsi[idx].values[0].x * h_evenPsi[idx].values[0].x + h_evenPsi[idx].values[0].y * h_evenPsi[idx].values[0].y;
					}
					sum *= unit;
					vol[idx] = (sum > 65535.0 ? 65535 : ushort(sum));
				}
			}
		}
		Text volpath;
		volpath << "volume" << iter << ".mhd";
		saveVolumeMap(volpath.str(), vol, dxsize, dysize, dzsize, block_scale * BLOCK_WIDTH);
#endif

		// finish iteration
		if (++iter > number_of_iterations) break;
		//++iter;
		//if (errorAbs > 0.01) break;

		// integrate one iteration
		std::cout << "Iteration " << iter << std::endl;
		for (uint step = 0; step < steps_per_iteration; step++)
		{
			// update odd values
			update << <dimGrid, dimBlock >> > (d_oddPsi, d_evenPsi, d_pot, d_lapind, d_hodges, c0, c1, dimensions);
			// update even values
			update << <dimGrid, dimBlock >> > (d_evenPsi, d_oddPsi, d_pot, d_lapind, d_hodges, c0, c1, dimensions);
		}
#if SAVE_PICTURE || SAVE_VOLUME
		// Copy back from device memory to host memory
		checkCudaErrors(cudaMemcpy3D(&evenPsiBackParams));
#endif
	}
	errorText.save("results/errors.txt");

	std::cout << "iteration time = " << (1e-3 * (clock() - time0)) / number_of_iterations << std::endl;
	std::cout << "total time = " << 1e-3 * (clock() - time0) << std::endl;

	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to launch kernels (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	return 0;
}

int main(int argc, char** argv)
{
#if LOAD_STATE_FROM_DISK
	VortexState state;
	state.load("state.dat");
	const ddouble eps = 1e-5 * state.searchFunctionMax();
	const ddouble maxr = state.searchMaxR(eps);
	const ddouble maxz = state.searchMaxZ(eps);
#else
	// preliminary vortex state to find vortex size
	VortexState state0;
	state0.setC0(C0);
	state0.setC1(C1);
	if(IS_3D) state0.setRange(0.0, 15.0, 35.0, 0.2, 0.2); // use this for 3d
	state0.iterateSolution(potentialRZ, 10000, 1e-29);
	const ddouble eps = 1e-5 * state0.searchFunctionMax();
	const ddouble minr = state0.searchMinR(eps);
	ddouble maxr = state0.searchMaxR(eps);
	ddouble maxz = state0.searchMaxZ(eps);
	//std::cout << "maxf=" << 1e6*eps << " minr=" << minr << " maxr=" << maxr << " maxz=" << maxz << std::endl;

	// more accurate vortex state
	VortexState state;
	state.setC0(C0);
	state.setC1(C1);
	if (IS_3D) state.setRange(minr, maxr, maxz, 0.03, 0.03); // use this for 3d
	state.initialize(state0);
	state.iterateSolution(potentialRZ, 10000, 1e-29);
	state.save("state.dat");
	maxr = state.searchMaxR(eps);
	maxz = state.searchMaxZ(eps);
	//std::cout << "maxf=" << state.searchFunctionMax() << std::endl;
#endif

	const int number_of_iterations = 100;
	const ddouble iteration_period = 1.0;
	const ddouble block_scale = PIx2 / (20.0 * sqrt(state.integrateCurvature()));

	std::cout << "1 GPU version" << std::endl;
	std::cout << "g = " << G << std::endl;
	std::cout << "ranks = 576" << std::endl;
	std::cout << "block_scale = " << block_scale << std::endl;
	std::cout << "iteration_period = " << iteration_period << std::endl;
	std::cout << "maxr = " << maxr << std::endl;
	std::cout << "maxz = " << maxz << std::endl;
	std::cout << "dual edge length = " << DUAL_EDGE_LENGTH * block_scale << std::endl;

	// integrate in time using DEC
	if (IS_3D) integrateInTime(state, block_scale, Vector3(-maxr, -maxr, -maxz), Vector3(maxr, maxr, maxz), iteration_period, number_of_iterations); // use this for 3d
	else integrateInTime(state, block_scale, Vector3(-maxr, -maxr, 0.0), Vector3(maxr, maxr, 0.0), iteration_period, number_of_iterations); // use this for 2d

	return 0;
}
