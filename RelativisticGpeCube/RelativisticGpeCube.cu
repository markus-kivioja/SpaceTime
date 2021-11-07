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
	double values[VALUES_IN_BLOCK];
};

struct BlockEdges
{
	double values[EDGES_IN_BLOCK];
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

__global__ void update_q(PitchedPtr next_q, PitchedPtr prev_q, PitchedPtr psi, int2* psiLapInd, uint3 dimensions, ddouble dtime_per_sigma, ddouble sign)
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
	double this_psi = ((BlockPsis*)pPsi)->values[0];

	size_t dualEdgeId = zid % EDGES_IN_BLOCK; // Dual node id. One thread per every dual node so VALUES_IN_BLOCK threads per mesh block (on z-axis)

	BlockEdges* prev = (BlockEdges*)(prev_q.ptr + prev_q.slicePitch * dataZid + prev_q.pitch * yid) + xid;
	BlockEdges* next = (BlockEdges*)(next_q.ptr + next_q.slicePitch * dataZid + next_q.pitch * yid) + xid;
	double d0_psi = ((BlockPsis*)(pPsi + psiLapInd[dualEdgeId + 3].x))->values[psiLapInd[dualEdgeId + 3].y] - this_psi;
	next->values[dualEdgeId] += sign * dtime_per_sigma * (prev->values[dualEdgeId] + d0_psi);
}

__global__ void update_psi(PitchedPtr nextStep, PitchedPtr prevStep, PitchedPtr qs, PitchedPtr potentials, int2* edgeLapInd, ddouble* hodges, ddouble g, uint3 dimensions, ddouble sign)
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
	double prev_psi = ((BlockPsis*)p_prev_psi)->values[dualNodeId];

	uint primaryFace = dualNodeId * FACE_COUNT;
	double d1_q = 0;
	d1_q -= hodges[0] * ((BlockEdges*)(p_qs + edgeLapInd[0].x))->values[2]; // -z
	d1_q -= hodges[1] * ((BlockEdges*)(p_qs + edgeLapInd[1].x))->values[0]; // -x
	d1_q -= hodges[2] * ((BlockEdges*)(p_qs + edgeLapInd[2].x))->values[1]; // -y
	d1_q += hodges[3] * ((BlockEdges*)p_qs)->values[0]; // +x
	d1_q += hodges[4] * ((BlockEdges*)p_qs)->values[1]; // +y
	d1_q += hodges[5] * ((BlockEdges*)p_qs)->values[2]; // +z

	double next_psi = p_next_psi->values[dualNodeId];

	double normsq = prev_psi * prev_psi + next_psi * next_psi;

	next_psi += sign * ((pot->values[dualNodeId] + g * normsq) * prev_psi - d1_q);

	p_next_psi->values[dualNodeId] = next_psi;
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
	std::cout << "bodies = " << xsize * ysize * zsize * bsize << std::endl;

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

	cudaPitchedPtr d_cudaR;
	cudaPitchedPtr d_cuda_q_R;
	cudaPitchedPtr d_cudaI;
	cudaPitchedPtr d_cuda_q_I;
	cudaPitchedPtr d_cudaPot;

	checkCudaErrors(cudaMalloc3D(&d_cudaR, psiExtent));
	checkCudaErrors(cudaMalloc3D(&d_cuda_q_R, edgeExtent));
	checkCudaErrors(cudaMalloc3D(&d_cudaI, psiExtent));
	checkCudaErrors(cudaMalloc3D(&d_cuda_q_I, edgeExtent));
	checkCudaErrors(cudaMalloc3D(&d_cudaPot, potExtent));

	size_t psiOffset = d_cudaR.pitch * dysize + d_cudaR.pitch + sizeof(BlockPsis);
	size_t edgeOffset = d_cuda_q_R.pitch * dysize + d_cuda_q_R.pitch + sizeof(BlockEdges);
	size_t potOffset = d_cudaPot.pitch * dysize + d_cudaPot.pitch + sizeof(BlockPots);
	PitchedPtr d_psiR = { (char*)d_cudaR.ptr + psiOffset, d_cudaR.pitch, d_cudaR.pitch * dysize };
	PitchedPtr d_qR = { (char*)d_cuda_q_R.ptr + edgeOffset, d_cuda_q_R.pitch, d_cuda_q_R.pitch * dysize };
	PitchedPtr d_psiI = { (char*)d_cudaI.ptr + psiOffset, d_cudaI.pitch, d_cudaI.pitch * dysize };
	PitchedPtr d_qI = { (char*)d_cuda_q_I.ptr + edgeOffset, d_cuda_q_I.pitch, d_cuda_q_I.pitch * dysize };
	PitchedPtr d_pot = { (char*)d_cudaPot.ptr + potOffset, d_cudaPot.pitch, d_cudaPot.pitch * dysize };

	// find terms for laplacian
	Buffer<int2> psiLapind;
	Buffer<int2> edgeLapind;
	Buffer<ddouble> hodges;
	ddouble lapfac = -0.5 * getLaplacian(psiLapind, hodges, sizeof(BlockPsis), d_psiR.pitch, d_psiR.slicePitch) / (block_scale * block_scale);
	getLaplacian(edgeLapind, hodges, sizeof(BlockEdges), d_qR.pitch, d_qR.slicePitch);
	const uint lapsize = psiLapind.size() / bsize;
	ddouble lapfac0 = lapsize * (-lapfac);

	//std::cout << "lapsize = " << lapsize << ", lapfac = " << lapfac << ", lapfac0 = " << lapfac0 << std::endl;

	// compute time step size
	const uint steps_per_iteration = 4 * uint(iteration_period * (maxpot + lapfac0)); // number of time steps per iteration period
	const ddouble dtime = iteration_period / ddouble(steps_per_iteration); // time step in time units
	const ddouble dtime_per_sigma = dtime / sigma;

	std::cout << "steps_per_iteration = " << steps_per_iteration << std::endl;

	std::cout << "ALU operations per unit time = " << xsize * ysize * zsize * bsize * steps_per_iteration * FACE_COUNT << std::endl;

	// multiply terms with dtime
	g *= dtime;
	lapfac *= dtime;
	lapfac0 *= dtime;
	for (i = 0; i < vsize; i++) pot[i] *= dtime;
	for (int i = 0; i < hodges.size(); ++i) hodges[i] = -0.5 * hodges[i] / (block_scale * block_scale) * dtime;

	int2* d_psiLapind;
	checkCudaErrors(cudaMalloc(&d_psiLapind, psiLapind.size() * sizeof(int2)));

	int2* d_edgeLapind;
	checkCudaErrors(cudaMalloc(&d_edgeLapind, edgeLapind.size() * sizeof(int2)));

	ddouble* d_hodges;
	checkCudaErrors(cudaMalloc(&d_hodges, hodges.size() * sizeof(ddouble)));

	// Initialize host memory
	size_t hostSize = dxsize * dysize * (zsize + 2);
	BlockPsis* h_r;// = new BlockPsis[dxsize * dysize * (zsize + 2)];
	BlockPsis* h_i;// = new BlockPsis[dxsize * dysize * (zsize + 2)];
	BlockPots* h_pot;// = new BlockPots[dxsize * dysize * (zsize + 2)];
	checkCudaErrors(cudaMallocHost(&h_r, hostSize * sizeof(BlockPsis)));
	checkCudaErrors(cudaMallocHost(&h_i, hostSize * sizeof(BlockPsis)));
	checkCudaErrors(cudaMallocHost(&h_pot, hostSize * sizeof(BlockPots)));
	memset(h_r, 0, hostSize * sizeof(BlockPsis));
	memset(h_i, 0, hostSize * sizeof(BlockPsis));
	memset(h_pot, 0, hostSize * sizeof(BlockPots));

	// initialize discrete field
	const Complex oddPhase = state.getPhase(-1 * dtime);
	Random rnd(54363);
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
					h_r[dstI].values[l] = r;
					h_i[dstI].values[l] = i;
					h_pot[dstI].values[l] = pot[srcI];
				}
			}
		}
	}

	cudaPitchedPtr h_cudaR = { 0 };
	cudaPitchedPtr h_cudaI = { 0 };
	cudaPitchedPtr h_cudaPot = { 0 };

	h_cudaR.ptr = h_r;
	h_cudaR.pitch = dxsize * sizeof(BlockPsis);
	h_cudaR.xsize = d_cudaR.xsize;
	h_cudaR.ysize = d_cudaR.ysize;

	h_cudaI.ptr = h_i;
	h_cudaI.pitch = dxsize * sizeof(BlockPsis);
	h_cudaI.xsize = d_cudaI.xsize;
	h_cudaI.ysize = d_cudaI.ysize;

	h_cudaPot.ptr = h_pot;
	h_cudaPot.pitch = dxsize * sizeof(BlockPots);
	h_cudaPot.xsize = d_cudaPot.xsize;
	h_cudaPot.ysize = d_cudaPot.ysize;

	// Copy from host memory to device memory
	cudaMemcpy3DParms rParams = { 0 };
	cudaMemcpy3DParms iParams = { 0 };
	cudaMemcpy3DParms potParams = { 0 };

	rParams.srcPtr = h_cudaR;
	rParams.dstPtr = d_cudaR;
	rParams.extent = psiExtent;
	rParams.kind = cudaMemcpyHostToDevice;

	iParams.srcPtr = h_cudaI;
	iParams.dstPtr = d_cudaI;
	iParams.extent = psiExtent;
	iParams.kind = cudaMemcpyHostToDevice;

	potParams.srcPtr = h_cudaPot;
	potParams.dstPtr = d_cudaPot;
	potParams.extent = potExtent;
	potParams.kind = cudaMemcpyHostToDevice;

	checkCudaErrors(cudaMemcpy3D(&rParams));
	checkCudaErrors(cudaMemcpy3D(&iParams));
	checkCudaErrors(cudaMemcpy3D(&potParams));
	checkCudaErrors(cudaMemcpy(d_psiLapind, &psiLapind[0], psiLapind.size() * sizeof(int2), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_edgeLapind, &edgeLapind[0], edgeLapind.size() * sizeof(int2), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_hodges, &hodges[0], hodges.size() * sizeof(ddouble), cudaMemcpyHostToDevice));

	// Clear host memory after data has been copied to devices
	cudaDeviceSynchronize();
	//Psi0.clear();
	pot.clear();
	bpos.clear();
	psiLapind.clear();
	edgeLapind.clear();
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
	cudaMemcpy3DParms rBackParams = { 0 };
	rBackParams.srcPtr = d_cudaR;
	rBackParams.dstPtr = h_cudaR;
	rBackParams.extent = psiExtent;
	rBackParams.kind = cudaMemcpyDeviceToHost;

	cudaMemcpy3DParms iBackParams = { 0 };
	iBackParams.srcPtr = d_cudaI;
	iBackParams.dstPtr = h_cudaI;
	iBackParams.extent = psiExtent;
	iBackParams.kind = cudaMemcpyDeviceToHost;
#endif
	Text errorText;
	const uint time0 = clock();
	const ddouble volume = (IS_3D ? block_scale : 1.0) * block_scale * block_scale * VOLUME;
	update_q << <edgeDimGrid, dimBlock >> > (d_qR, d_qR, d_psiR, d_psiLapind, dimensions, dtime_per_sigma / (dtime - dtime_per_sigma), 1.0f);
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
				double norm = sqrt(h_r[idx].values[0] * h_r[idx].values[0] + h_i[idx].values[0] * h_i[idx].values[0]);
		
				pic.setColor(i, j, INTENSITY * Vector4(h_r[idx].values[0], norm, h_i[idx].values[0], 1.0));
			}
		}
		std::ostringstream picpath;
		picpath << "results/kuva" << iter << ".bmp";
		pic.save(picpath.str(), false);

		// print squared norm and error
		const Complex currentPhase = state.getPhase(iter * steps_per_iteration * dtime);
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

						Complex evenPsi(h_r[dstI].values[l], h_i[dstI].values[l]);
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
			// update odd values (imaginary terms)
			update_psi << <psiDimGrid, dimBlock >> > (d_psiI, d_psiR, d_qR, d_pot, d_edgeLapind, d_hodges, g, dimensions, -1.0f);
			update_q << <edgeDimGrid, dimBlock >> > (d_qI, d_qR, d_psiR, d_psiLapind, dimensions, dtime_per_sigma, 1.0f);
			// update even values (real terms)
			update_psi << <psiDimGrid, dimBlock >> > (d_psiR, d_psiI, d_qI, d_pot, d_edgeLapind, d_hodges, g, dimensions, 1.0f);
			update_q << <edgeDimGrid, dimBlock >> > (d_qR, d_qI, d_psiI, d_psiLapind, dimensions, dtime_per_sigma, -1.0f);
		}
#if SAVE_PICTURE || SAVE_VOLUME
		// Copy back from device memory to host memory
		checkCudaErrors(cudaMemcpy3D(&rBackParams));
		checkCudaErrors(cudaMemcpy3D(&iBackParams));
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
	if(IS_3D) state0.setRange(0.0, 15.0, 35.0, 0.2, 0.2); // use this for 3d
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
	const ddouble sigma = 0.001;
	integrateInTime(state, block_scale, Vector3(-maxr, -maxr, -maxz), Vector3(maxr, maxr, maxz), iteration_period, number_of_iterations, sigma);

	return 0;
}
