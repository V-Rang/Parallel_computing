/*

Parallel reduction for n = 2^16 length array of ones.
call 8 blocks of 32 threads each. (i.e. total = 256 threads) 
Each thread calculates sum of 256 elements (256 X 256 = 2^16) and stores answer in "result" array in element "result[tid]".
Call single thread to sum the 256 elements of "result"
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

__global__ void summer(int *a, int size,int *result)
{
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	result[tid] = 0;
	for(int x = tid*256; x < 256*(tid+1); x++) result[tid] += a[x];
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
	summer<<<8,32>>>(d_x,n,result_1);

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
	summer<<<1,1>>>(result_1,256,d_final_answer);
	cudaMemcpy(h_final_answer,d_final_answer,1*sizeof(int),cudaMemcpyDeviceToHost);
	cout << h_final_answer[0] << endl;

	return 0;
}