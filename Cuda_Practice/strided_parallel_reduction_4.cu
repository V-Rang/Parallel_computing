/*
Strided Parallel reduction for n = 2^16 length array of ones.
Working:
Call 128 blocks, each containing 256 threads.
assign a shared array "partial_sum" for all threads within a block.
Partial sum for each block is calculated as sum of 256 elements i.e.

if global array has 16 elements -> 0 ... 15, instead of using 4 blocks of 4 threads each, we use 2 blocks of 4 threads each
where:
parital sums array for block 0 -> (0th + 4th) ,(1st + 5th), (2nd + 6th), (3rd + 7th).
partial sums array for block 1 -> (8th + 12th) ,(9th + 13th), (10th + 14th), (11th + 15th).

Then the result[bIdx.x] = ps[0] for each block.

Call sum function again using 1 block of 64 threads for calculating sum of the 128 elements in the results array.
the partial sums array for this single block:
ps[0] = global[0] + global[64]
...
ps[63] = global[63] + global[127]
*/


#include<iostream>
#include<cuda.h>
#include<cuda_runtime.h>

using namespace std;

const int blocksize = 256; //no of threads in each block

__global__ void init(int *a, int size)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid < size) a[tid] = 1;
}


__global__ void summer_halved(int *d_x, int n, int *result_1)
{
    __shared__ int partial_sum[blocksize];
    int i = blockDim.x * 2 * blockIdx.x;
    partial_sum[threadIdx.x] = d_x[i] + d_x[i+blockDim.x];
    __syncthreads();

    for(int s = blockDim.x/2 ; s > 0 ; s >>= 1)
    {
        if(threadIdx.x < s) partial_sum[threadIdx.x] += partial_sum[threadIdx.x + s];
        __syncthreads();
    }

    result_1[blockIdx.x] = partial_sum[0];
}

int main()
{
    int n = 1 << 16;
    int *d_x;
    
    cudaMalloc(&d_x,n*sizeof(int));
    init<<<n/256,256>>>(d_x,n); 
    
    int *result_1;
    cudaMalloc(&result_1,(256/2)*sizeof(int));

    summer_halved<<<n/512,256>>>(d_x,n,result_1);
    cudaDeviceSynchronize();

    int *h_final_answer, *d_final_answer;
    h_final_answer = new int[1];
    cudaMalloc(&d_final_answer,1*sizeof(int));

    summer_halved<<<1,64>>>(result_1,n,d_final_answer);
    
    cudaMemcpy(h_final_answer,d_final_answer,1*sizeof(int),cudaMemcpyDeviceToHost);

    printf("Final_sum  = %d\n",h_final_answer[0]);

    return 0;
}