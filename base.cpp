// ----------------------------------------------------------

// ----------------------------------------------------------


#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cmath>

#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip> 
#include <chrono>
#include <stack>
using namespace std;

// ----------------------------------------------------------

// We provide a small library so we can easily setup OpenCL
#include "clutils.h"

// ----------------------------------------------------------

// ----------------------------------------------------------
int* mulMat(int*gridCpu,int* grid2Cpu,int N){
	int tmp = 0;
	int* newgridCpu = new int[N*N];
	for(int i = 0; i < N ;i++){
		for(int j = 0; j < N;j++){
			for(int k = 0;k < N;k++){
				tmp = tmp + (gridCpu[k*N+j] * grid2Cpu[i*N+k]);
			}
		newgridCpu[i*N+j] = tmp;
		tmp = 0;
		}
		}
return newgridCpu;
}
//The complexity of this algorithm would be about O(log_2(N³)) since we cut in half when we have an even k, and when we don't k becomes even on the next iteration
int* expoMat(int* gridCpu,int N,int K){
	if (K == 1){
		return gridCpu;
	}
	if(K%2 != 0){
		return mulMat(gridCpu,expoMat(gridCpu,N,K-1),N);
	} else{
		return mulMat(expoMat(gridCpu,N,K/2),expoMat(gridCpu,N,K/2),N);
	}
}
int* expoMatGPU(int* grid,int N,int K,cl::Kernel* kernel,cl::Buffer p,cl::Buffer q,cl::Buffer r){
	clu_Queue->enqueueReadBuffer(r, true, 0, (N * N) * sizeof(int),grid);

	if (K == 1){
		return grid;
	}
	if(K%2 != 0){
		clu_Queue->enqueueWriteBuffer(p, true, 0, (N * N) * sizeof(int), expoMatGPU(grid,N,K/2,kernel,p,q,r));
		clu_Queue->enqueueWriteBuffer(q, true, 0, (N * N) * sizeof(int), expoMatGPU(grid,N,K/2,kernel,p,q,r));	
	} else{
		clu_Queue->enqueueWriteBuffer(p, true, 0, (N * N) * sizeof(int), grid);
		clu_Queue->enqueueWriteBuffer(q, true, 0, (N * N) * sizeof(int), expoMatGPU(grid,N,K-1,kernel,p,q,r));	
	}
	//executing Kernel
		kernel->setArg(0, p);
		kernel->setArg(1, q);
		kernel->setArg(2, r);
		kernel->setArg(3, N);
		cl_int clerr = clu_Queue->enqueueNDRangeKernel(*kernel, cl::NullRange, cl::NDRange(N*N),cl::NullRange);
		cluCheckError(clerr, "Error running the kernel");
}
int main(int argc, char **argv)
{
	const char *clu_File = SRC_PATH "base.cl";  // path to file containing OpenCL kernel(s) code

	// Initialize OpenCL
	cluInit();

	// After this call you have access to
	// clu_Context;      <= OpenCL context (pointer)
	// clu_Devices;      <= OpenCL device list (vector)
	// clu_Queue;        <= OpenCL queue (pointer)

	// Load Program
	cl::Program *program = cluLoadProgram(clu_File);
	cl::Kernel *kernel = cluLoadKernel(program,"mulmat");

	// allocate memory and opencl buffers
	const int N = 2;
	cl::Buffer p_buffer(*clu_Context, CL_MEM_READ_ONLY, (N*N) * sizeof(int));
	cl::Buffer q_buffer(*clu_Context, CL_MEM_READ_ONLY,  (N*N) * sizeof(int));
	cl::Buffer r_buffer(*clu_Context, CL_MEM_WRITE_ONLY, (N*N) * sizeof(int));

	int* gridCpu = new int[N*N]{1,2,3,4};

	int* grid2Cpu = new int[N*N]{1,2,3,4};

	//algo cpu
	
	int* newgridCpu = mulMat(gridCpu,grid2Cpu,N);
	
	std::cout << "Mulmat CPU" << endl;
	std::cout << std::endl;

	for(int i = 0; i < N ;i++){
		for(int j = i; j < N*N;j+=N){
		
		std::cout << newgridCpu[j] << " ";

		}
		std::cout << endl;

		}

	std::cout << "Expomat CPU" << endl;
	int K = 6;
	newgridCpu = expoMat(gridCpu,N,K);
	std::cout << std::endl;
	
	for(int i = 0; i < N ;i++){
		for(int j = i; j < N*N;j+=N){
		
		std::cout << newgridCpu[j] << " ";

		}
		std::cout << endl;

		}


	std::cout << "Mulmat GPU" << endl;

	std::cout << std::endl;


	int grid[N* N] = {1,2,3,4};
	int res[N][N];

	clu_Queue->enqueueWriteBuffer(p_buffer, true, 0, (N * N) * sizeof(int), grid);
	clu_Queue->enqueueWriteBuffer(q_buffer, true, 0, (N * N) * sizeof(int), grid);

	//executing Kernel
	kernel->setArg(0, p_buffer);
	kernel->setArg(1, q_buffer);
	kernel->setArg(2, r_buffer);
	kernel->setArg(3, N);

	cl_int clerr = clu_Queue->enqueueNDRangeKernel(*kernel, cl::NullRange, cl::NDRange(N*N),cl::NullRange);
	cluCheckError(clerr, "Error running the kernel");
	
	clu_Queue->enqueueReadBuffer(r_buffer, true, 0, (N * N) * sizeof(int),res);


	for(int i = 0; i < N ;i++){
		for(int j = 0; j < N;j++){
		
		std::cout << res[j][i] << " ";

		}
		std::cout << endl;

		}
	std::cout << "Expomat GPU" << std::endl;
		std::cout << std::endl;

	int* resMat = expoMatGPU(grid,N,K,kernel,p_buffer,q_buffer,r_buffer);
	for(int i = 0; i < N ;i++){
		for(int j = i; j < N*N;j+=N){
		
		
		std::cout << resMat[j] << " ";

		}
		std::cout << endl;

		}
	std::cout << std::endl;
}