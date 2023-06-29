/*
Strided Parallel reduction for n = 2^16 length array of ones.
Working:
Call 256 blocks, each containing 256 threads.
assign a shared array "partial_sum" for all threads within a block.
All the 256 values corresponding to a given a block are copied into this partial array.
A strided sum is performed within this "partial_sum" array.
ps[0] contains the sum of all elements corresponding to a particular block, this is copied onto a "result array".

the above step is repeated once again for the "result array" where a single block of 256 threads is called to 
sum the 256 elements in the "result array".
*/


#include<iostream>
#include<cublas.h>
#include<cuda.h>
#include<cuda_runtime.h>

using namespace std;

__global__ void init(int *a, int size)
{
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	if(tid < size) a[tid] = 1;
}


const int bsize = 256;
__global__ void summer(int *a,int *result)
{
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	__shared__ int partial_sum[bsize];

	//copy values from block into partial sum array
	partial_sum[threadIdx.x] = a[tid];
	__syncthreads();

	//strided sum within partial sum array to get sum of the array into its 0th element
	for(int s = 1; s< blockDim.x ; s *= 2)
	{
		if(threadIdx.x % (2*s) == 0) partial_sum[threadIdx.x] += partial_sum[threadIdx.x + s];
		__syncthreads(); //neeeded to ensure every thread within block is done doing its work before going on to next iteration.

	}  
	result[blockIdx.x] = partial_sum[0];

}

int main()
{
	int n = 1 << 16;
	int *x = new int[n];

	int *d_x;
	cudaMalloc(&d_x,n*sizeof(int));
	init<<<n/256,256>>>(d_x,n);

	int *result_1;
	cudaMalloc(&result_1,256*sizeof(int));
	summer<<<n/256,256>>>(d_x,result_1);
	cudaDeviceSynchronize();

	int *h_result1 = new int[256];
	cudaMemcpy(h_result1,result_1,256*sizeof(int),cudaMemcpyDeviceToHost);

	//onCPU:
	// int final_answer = 0;
	// for(int i=0;i<256;i++) final_answer += h_result1[i];
	// cout << final_answer << endl;

	//onGPU
	// int *h_final_answer;// incorrect, need to allocate host memory to which to copy the final answer from device
	int *h_final_answer = new int[1];
	int *d_final_answer;
	cudaMalloc(&d_final_answer,1*sizeof(int));
	summer<<<1,256>>>(result_1,d_final_answer);
	cudaMemcpy(h_final_answer,d_final_answer,1*sizeof(int),cudaMemcpyDeviceToHost);
	cout << h_final_answer[0] << endl;

	return 0;
}