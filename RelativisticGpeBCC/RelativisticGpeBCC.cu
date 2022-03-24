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

ddouble RATIO = 1.0;
ddouble KAPPA = 10;
ddouble G = 300;

#define RELATIVISTIC 1

#define LOAD_STATE_FROM_DISK 1
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

struct BlockPsis
{
	double2 values[VALUES_IN_BLOCK];
};

struct BlockEdges
{
	double2 values[EDGES_IN_BLOCK];
};

struct BlockPots
{
	double values[VALUES_IN_BLOCK];
};

struct PitchedPtr
{
	char* ptr;
	size_t pitch;
	size_t slicePitch;
};

__global__ void update_q(PitchedPtr next_q, PitchedPtr prev_q, PitchedPtr psi, int3* d0, uint3 dimensions, ddouble dtime_per_sigma)
{
	size_t xid = blockIdx.x * blockDim.x + threadIdx.x;
	size_t yid = blockIdx.y * blockDim.y + threadIdx.y;
	size_t zid = blockIdx.z * blockDim.z + threadIdx.z;

	size_t dataZid = zid / EDGES_IN_BLOCK; // One thread per every dual edge so EDGES_IN_BLOCK threads per mesh block (on z-axis)

	// Exit leftover threads
	if (xid >= dimensions.x || yid >= dimensions.y || dataZid >= dimensions.z)
	{
		return;
	}

	char* pPsi = psi.ptr + psi.slicePitch * dataZid + psi.pitch * yid + sizeof(BlockPsis) * xid;

	size_t dualEdgeId = zid % EDGES_IN_BLOCK; // Dual node id. One thread per every dual node so VALUES_IN_BLOCK threads per mesh block (on z-axis)

	BlockEdges* next = (BlockEdges*)(next_q.ptr + next_q.slicePitch * dataZid + next_q.pitch * yid) + xid;
	double2 d0psi = ((BlockPsis*)(pPsi + d0[dualEdgeId].y))->values[d0[dualEdgeId].z] -
		((BlockPsis*)(pPsi))->values[d0[dualEdgeId].x];

#if RELATIVISTIC
	BlockEdges* prev = (BlockEdges*)(prev_q.ptr + prev_q.slicePitch * dataZid + prev_q.pitch * yid) + xid;
	double2 q = dtime_per_sigma * (prev->values[dualEdgeId] + d0psi);
	next->values[dualEdgeId] += make_double2(-q.y, q.x);
#else
	next->values[dualEdgeId] = d0psi;
#endif
}

__global__ void update_psi(PitchedPtr nextStep, PitchedPtr prevStep, PitchedPtr qs, PitchedPtr potentials, int2* d1, ddouble* hodges, ddouble g, uint3 dimensions)
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
	char* p_prev_psi = prevStep.ptr + prevStep.slicePitch * dataZid + prevStep.pitch * yid + sizeof(BlockPsis) * xid;
	char* p_qs = qs.ptr + qs.slicePitch * dataZid + qs.pitch * yid + sizeof(BlockEdges) * xid;
	BlockPsis* p_next_psi = (BlockPsis*)(nextStep.ptr + nextStep.slicePitch * dataZid + nextStep.pitch * yid) + xid;
	BlockPots* pot = (BlockPots*)(potentials.ptr + potentials.slicePitch * dataZid + potentials.pitch * yid) + xid;

	// Update psi
	size_t dualNodeId = zid % VALUES_IN_BLOCK; // Dual node id. One thread per every dual node so VALUES_IN_BLOCK threads per mesh block (on z-axis)
	double2 prev_psi = ((BlockPsis*)p_prev_psi)->values[dualNodeId];

	uint startEdgeId = dualNodeId * FACE_COUNT;
	double2 d1q = make_double2(0, 0);
	for (int edgeIdOffset = 0; edgeIdOffset < FACE_COUNT; ++edgeIdOffset)
	{
		int edgeId = startEdgeId + edgeIdOffset;
		d1q += hodges[edgeId] * ((BlockEdges*)(p_qs + d1[edgeId].x))->values[d1[edgeId].y];
	}


	double normsq = (conj(prev_psi) * prev_psi).x;

#if RELATIVISTIC
	double2 Hpsi = (pot->values[dualNodeId] + g * normsq) * prev_psi - d1q;
#else
	double2 Hpsi = d1q + (pot->values[dualNodeId] + g * normsq) * prev_psi;
#endif
	p_next_psi->values[dualNodeId] += make_double2(Hpsi.y, -Hpsi.x);
};

uint integrateInTime(const VortexState& state, const ddouble block_scale, const Vector3& minp, const Vector3& maxp, const ddouble iteration_period, const uint number_of_iterations, ddouble sigma)
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
	ddouble g = state.getG(); // effective interaction strength
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
	cudaExtent edgeExtent = make_cudaExtent(dxsize * sizeof(BlockEdges), dysize, dzsize);
	cudaExtent potExtent = make_cudaExtent(dxsize * sizeof(BlockPots), dysize, dzsize);

	cudaPitchedPtr d_cudaEven;
	cudaPitchedPtr d_cuda_q_even;
	cudaPitchedPtr d_cudaOdd;
	cudaPitchedPtr d_cuda_q_odd;
	cudaPitchedPtr d_cudaPot;

	checkCudaErrors(cudaMalloc3D(&d_cudaEven, psiExtent));
	checkCudaErrors(cudaMalloc3D(&d_cuda_q_even, edgeExtent));
	checkCudaErrors(cudaMalloc3D(&d_cudaOdd, psiExtent));
	checkCudaErrors(cudaMalloc3D(&d_cuda_q_odd, edgeExtent));
	checkCudaErrors(cudaMalloc3D(&d_cudaPot, potExtent));

	size_t psiOffset = d_cudaEven.pitch * dysize + d_cudaEven.pitch + sizeof(BlockPsis);
	size_t edgeOffset = d_cuda_q_even.pitch * dysize + d_cuda_q_even.pitch + sizeof(BlockEdges);
	size_t potOffset = d_cudaPot.pitch * dysize + d_cudaPot.pitch + sizeof(BlockPots);
	PitchedPtr d_psiEven = { (char*)d_cudaEven.ptr + psiOffset, d_cudaEven.pitch, d_cudaEven.pitch * dysize };
	PitchedPtr d_qEven = { (char*)d_cuda_q_even.ptr + edgeOffset, d_cuda_q_even.pitch, d_cuda_q_even.pitch * dysize };
	PitchedPtr d_psiOdd = { (char*)d_cudaOdd.ptr + psiOffset, d_cudaOdd.pitch, d_cudaOdd.pitch * dysize };
	PitchedPtr d_qOdd = { (char*)d_cuda_q_odd.ptr + edgeOffset, d_cuda_q_odd.pitch, d_cuda_q_odd.pitch * dysize };
	PitchedPtr d_pot = { (char*)d_cudaPot.ptr + potOffset, d_cudaPot.pitch, d_cudaPot.pitch * dysize };

	// find terms for laplacian
	Buffer<int3> d0;
	Buffer<int2> d1;
	Buffer<ddouble> hodges;
	ddouble lapfac = -0.5 * getLaplacian(hodges, d0, d1, sizeof(BlockPsis), d_psiEven.pitch, d_psiEven.slicePitch, sizeof(BlockEdges), d_qEven.pitch, d_qEven.slicePitch) / (block_scale * block_scale);
	const uint lapsize = hodges.size() / bsize;
	ddouble lapfac0 = lapsize * (-lapfac);

	//std::cout << "lapsize = " << lapsize << ", lapfac = " << lapfac << ", lapfac0 = " << lapfac0 << std::endl;

	// compute time step size
	//const uint steps_per_iteration = uint(iteration_period * (maxpot + lapfac0)) + 1;
#if RELATIVISTIC
	const uint steps_per_iteration = 185; // 185 is the min stable for hyperbolic
#else
	const uint steps_per_iteration = 522; // ... and 522 for the parabolic
#endif
	const ddouble dt = iteration_period / ddouble(steps_per_iteration); // time step in time units
	const ddouble dt_per_sigma = dt / sigma;

	std::cout << "steps_per_iteration = " << steps_per_iteration << std::endl;

	std::cout << "ALU operations per unit time = " << bodies * steps_per_iteration * FACE_COUNT << std::endl;
	std::cout << "Domain: " << domain.x << ", " << domain.y << ", " << domain.z << std::endl;
	std::cout << "Maxpot:" << maxpot << std::endl;

	// multiply terms with dt
	g *= dt;
	lapfac *= dt;
	lapfac0 *= dt;
	for (i = 0; i < vsize; i++) pot[i] *= dt;
	for (int i = 0; i < hodges.size(); ++i) hodges[i] = -0.5 * hodges[i] / (block_scale * block_scale) * dt;

	int3* d_d0;
	checkCudaErrors(cudaMalloc(&d_d0, d0.size() * sizeof(int3)));

	int2* d_d1;
	checkCudaErrors(cudaMalloc(&d_d1, d1.size() * sizeof(int2)));

	ddouble* d_hodges;
	checkCudaErrors(cudaMalloc(&d_hodges, hodges.size() * sizeof(ddouble)));

	// Initialize host memory
	size_t hostSize = dxsize * dysize * (zsize + 2);
	BlockPsis* h_even;// = new BlockPsis[dxsize * dysize * (zsize + 2)];
	BlockPsis* h_odd;// = new BlockPsis[dxsize * dysize * (zsize + 2)];
	BlockPots* h_pot;// = new BlockPots[dxsize * dysize * (zsize + 2)];
	checkCudaErrors(cudaMallocHost(&h_even, hostSize * sizeof(BlockPsis)));
	checkCudaErrors(cudaMallocHost(&h_odd, hostSize * sizeof(BlockPsis)));
	checkCudaErrors(cudaMallocHost(&h_pot, hostSize * sizeof(BlockPots)));
	memset(h_even, 0, hostSize * sizeof(BlockPsis));
	memset(h_odd, 0, hostSize * sizeof(BlockPsis));
	memset(h_pot, 0, hostSize * sizeof(BlockPots));

	// initialize discrete field
	const Complex oddPhase = state.getPhase(-0.5 * dt);
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
					ddouble r = Psi0[srcI].r;
					ddouble i = Psi0[srcI].i;
					double2 even = make_double2(Psi0[srcI].r, Psi0[srcI].i); // No noice
					h_even[dstI].values[l] = even;
					h_odd[dstI].values[l] = make_double2(oddPhase.r * even.x - oddPhase.i * even.y,
						oddPhase.i * even.x + oddPhase.r * even.y);
					h_pot[dstI].values[l] = pot[srcI];
				}
			}
		}
	}

	cudaPitchedPtr h_cudaEven = { 0 };
	cudaPitchedPtr h_cudaOdd = { 0 };
	cudaPitchedPtr h_cudaPot = { 0 };

	h_cudaEven.ptr = h_even;
	h_cudaEven.pitch = dxsize * sizeof(BlockPsis);
	h_cudaEven.xsize = d_cudaEven.xsize;
	h_cudaEven.ysize = d_cudaEven.ysize;

	h_cudaOdd.ptr = h_odd;
	h_cudaOdd.pitch = dxsize * sizeof(BlockPsis);
	h_cudaOdd.xsize = d_cudaOdd.xsize;
	h_cudaOdd.ysize = d_cudaOdd.ysize;

	h_cudaPot.ptr = h_pot;
	h_cudaPot.pitch = dxsize * sizeof(BlockPots);
	h_cudaPot.xsize = d_cudaPot.xsize;
	h_cudaPot.ysize = d_cudaPot.ysize;

	// Copy from host memory to device memory
	cudaMemcpy3DParms evenParams = { 0 };
	cudaMemcpy3DParms oddParams = { 0 };
	cudaMemcpy3DParms potParams = { 0 };

	evenParams.srcPtr = h_cudaEven;
	evenParams.dstPtr = d_cudaEven;
	evenParams.extent = psiExtent;
	evenParams.kind = cudaMemcpyHostToDevice;

	oddParams.srcPtr = h_cudaOdd;
	oddParams.dstPtr = d_cudaOdd;
	oddParams.extent = psiExtent;
	oddParams.kind = cudaMemcpyHostToDevice;

	potParams.srcPtr = h_cudaPot;
	potParams.dstPtr = d_cudaPot;
	potParams.extent = potExtent;
	potParams.kind = cudaMemcpyHostToDevice;

	checkCudaErrors(cudaMemcpy3D(&evenParams));
	checkCudaErrors(cudaMemcpy3D(&oddParams));
	checkCudaErrors(cudaMemcpy3D(&potParams));
	checkCudaErrors(cudaMemcpy(d_d0, &d0[0], d0.size() * sizeof(int3), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_d1, &d1[0], d1.size() * sizeof(int2), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_hodges, &hodges[0], hodges.size() * sizeof(ddouble), cudaMemcpyHostToDevice));

	// Clear host memory after data has been copied to devices
	cudaDeviceSynchronize();
	//Psi0.clear();
	pot.clear();
	bpos.clear();
	d0.clear();
	d1.clear();
	hodges.clear();
	cudaFreeHost(h_pot);
#if !(SAVE_PICTURE || SAVE_VOLUME)
	cudaFreeHost(h_evenPsi);
#endif

	// Integrate in time
	uint3 dimensions = make_uint3(xsize, ysize, zsize);
	uint iter = 0;
	dim3 dimBlock(THREAD_BLOCK_X, THREAD_BLOCK_Y, THREAD_BLOCK_Z);
	dim3 psiDimGrid((xsize + THREAD_BLOCK_X - 1) / THREAD_BLOCK_X,
		(ysize + THREAD_BLOCK_Y - 1) / THREAD_BLOCK_Y,
		((zsize + THREAD_BLOCK_Z - 1) / THREAD_BLOCK_Z) * VALUES_IN_BLOCK);
	dim3 edgeDimGrid((xsize + THREAD_BLOCK_X - 1) / THREAD_BLOCK_X,
		(ysize + THREAD_BLOCK_Y - 1) / THREAD_BLOCK_Y,
		((zsize + THREAD_BLOCK_Z - 1) / THREAD_BLOCK_Z) * EDGES_IN_BLOCK);
#if SAVE_PICTURE || SAVE_VOLUME
	cudaMemcpy3DParms evenBackParams = { 0 };
	evenBackParams.srcPtr = d_cudaEven;
	evenBackParams.dstPtr = h_cudaEven;
	evenBackParams.extent = psiExtent;
	evenBackParams.kind = cudaMemcpyDeviceToHost;

	cudaMemcpy3DParms oddBackParams = { 0 };
	oddBackParams.srcPtr = d_cudaOdd;
	oddBackParams.dstPtr = h_cudaOdd;
	oddBackParams.extent = psiExtent;
	oddBackParams.kind = cudaMemcpyDeviceToHost;
#endif
	Text errorText;
	const uint time0 = clock();
	const ddouble volume = (IS_3D ? block_scale : 1.0) * block_scale * block_scale * VOLUME;
#if RELATIVISTIC
	update_q << <edgeDimGrid, dimBlock >> > (d_qEven, d_qEven, d_psiEven, d_d0, dimensions, dt_per_sigma / (dt - dt_per_sigma));
#endif
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
				double norm = sqrt(h_even[idx].values[0].x * h_even[idx].values[0].x + h_even[idx].values[0].y * h_even[idx].values[0].y);

				pic.setColor(i, j, INTENSITY * Vector4(h_even[idx].values[0].x, norm, h_even[idx].values[0].y, 1.0));
			}
		}
		std::ostringstream picpath;
#if RELATIVISTIC
		picpath << "results/hyper_" << iter << ".bmp";
#else
		picpath << "results/para_" << iter << ".bmp";
#endif
		pic.save(picpath.str(), false);

		// print squared norm and error
		const Complex currentPhase = state.getPhase(iter * steps_per_iteration * dt);
		ddouble errorNormSq = 0;
		ddouble normsq = 0.0;
		Complex error(0.0, 0.0);
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

						Complex evenPsi(h_even[dstI].values[l].x, h_even[dstI].values[l].y);
						normsq += evenPsi.normsq() * volume;
						error += (Psi0[srcI].con() * evenPsi) * volume;

						Complex groundTruth = currentPhase * Psi0[srcI];
						errorNormSq += (groundTruth - evenPsi).normsq();
					}
				}
			}
		}
		ddouble RMSE = sqrt(errorNormSq / (double)(zsize * ysize * xsize * bsize));
		ddouble errorAbs = abs(normsq - error.norm());
		std::cout << "normsq=" << normsq << " error=" << errorAbs << std::endl;
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
						sum += h_r[idx].values[0] * h_r[idx].values[0] + h_evenPsi[idx].values[0].y * h_evenPsi[idx].values[0].y;
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
#if RELATIVISTIC
			// update odd values (imaginary terms)
			update_psi << <psiDimGrid, dimBlock >> > (d_psiOdd, d_psiEven, d_qEven, d_pot, d_d1, d_hodges, g, dimensions);
			update_q << <edgeDimGrid, dimBlock >> > (d_qOdd, d_qEven, d_psiEven, d_d0, dimensions, dt_per_sigma);
			// update even values (real terms)
			update_psi << <psiDimGrid, dimBlock >> > (d_psiEven, d_psiOdd, d_qOdd, d_pot, d_d1, d_hodges, g, dimensions);
			update_q << <edgeDimGrid, dimBlock >> > (d_qEven, d_qOdd, d_psiOdd, d_d0, dimensions, dt_per_sigma);
#else
			// update odd values (imaginary terms)
			update_q << <edgeDimGrid, dimBlock >> > (d_qEven, d_qEven, d_psiEven, d_d0, dimensions, dt_per_sigma);
			update_psi << <psiDimGrid, dimBlock >> > (d_psiOdd, d_psiEven, d_qEven, d_pot, d_d1, d_hodges, g, dimensions);
			// update even values (real terms)
			update_q << <edgeDimGrid, dimBlock >> > (d_qOdd, d_qOdd, d_psiOdd, d_d0, dimensions, dt_per_sigma);
			update_psi << <psiDimGrid, dimBlock >> > (d_psiEven, d_psiOdd, d_qOdd, d_pot, d_d1, d_hodges, g, dimensions);
#endif
		}
#if SAVE_PICTURE || SAVE_VOLUME
		// Copy back from device memory to host memory
		checkCudaErrors(cudaMemcpy3D(&evenBackParams));
		checkCudaErrors(cudaMemcpy3D(&oddBackParams));
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
	state0.setKappa(KAPPA);
	state0.setG(G);
	if (IS_3D) state0.setRange(0.0, 15.0, 35.0, 0.2, 0.2); // use this for 3d
	state0.iterateSolution(potentialRZ, 10000, 1e-29);
	const ddouble eps = 1e-5 * state0.searchFunctionMax();
	const ddouble minr = state0.searchMinR(eps);
	ddouble maxr = state0.searchMaxR(eps);
	ddouble maxz = state0.searchMaxZ(eps);
	//std::cout << "maxf=" << 1e6*eps << " minr=" << minr << " maxr=" << maxr << " maxz=" << maxz << std::endl;

	// more accurate vortex state
	VortexState state;
	state.setKappa(KAPPA);
	state.setG(G);
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
	std::cout << "kappa = " << KAPPA << std::endl;
	std::cout << "g = " << G << std::endl;
	std::cout << "ranks = 576" << std::endl;
	std::cout << "block_scale = " << block_scale << std::endl;
	std::cout << "iteration_period = " << iteration_period << std::endl;
	std::cout << "maxr = " << maxr << std::endl;
	std::cout << "maxz = " << maxz << std::endl;
	std::cout << "dual edge length = " << DUAL_EDGE_LENGTH * block_scale << std::endl;

	// Integrate in time using DEC
	const ddouble sigma = 0.01;
	integrateInTime(state, block_scale, Vector3(-maxr, -maxr, -maxz), Vector3(maxr, maxr, maxz), iteration_period, number_of_iterations, sigma);

	return 0;
}
