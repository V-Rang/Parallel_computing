/*
Using device function to speed up to compensate for the fact that in sum_reduction, only half the threads are active 
i.e. 128 -> 64 -> 32 -> 16 -> 8 -> 4 -> 2 -> 1 for a block of length 256.
 
*/

#include<iostream>
#include<cuda.h>
#include<cuda_runtime.h>

using namespace std;

const int blocksize = 256; //no of threads in each block

__global__ void init(int *x, int n)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid < n) x[tid] = 1;
}

__device__ void warpReduce(volatile int *partial_sum, int t)
{
    partial_sum[t] += partial_sum[t+32];
    partial_sum[t] += partial_sum[t+16];
    partial_sum[t] += partial_sum[t+8];
    partial_sum[t] += partial_sum[t+4];
    partial_sum[t] += partial_sum[t+2];
    partial_sum[t] += partial_sum[t+1];
    
}

__global__ void sum_reduction(int *x, int *result)
{
    __shared__ int partial_sum[blocksize];
    int i = blockDim.x * 2 * blockIdx.x;

    partial_sum[threadIdx.x] = x[i] + x[i+blockDim.x];
    __syncthreads();

    for(int s = blockDim.x/2 ; s > 32; s >>= 1)
    {
        if(threadIdx.x < s)
        partial_sum[threadIdx.x] += partial_sum[threadIdx.x + s];

        __syncthreads();
    }
    if(threadIdx.x < 32) warpReduce(partial_sum,threadIdx.x);
    __syncthreads();

    if(threadIdx.x == 0)
    result[blockIdx.x] = partial_sum[0];

}
int main()
{
    int *d_x;

    int n = 1 << 16;

    cudaMalloc(&d_x,n*sizeof(int));

    init<<<256,256>>>(d_x,n);

    int *intermediate_result;
    cudaMalloc(&intermediate_result,(n/512)*sizeof(int));
    sum_reduction<<<128,256>>>(d_x,intermediate_result);

    int *h_final_result = new int[1];
    int *d_final_result;
    cudaMalloc(&d_final_result,sizeof(int));
    sum_reduction<<<1,64>>>(intermediate_result,d_final_result);

    cudaMemcpy(h_final_result,d_final_result,1*sizeof(int),cudaMemcpyDeviceToHost);

    cout << h_final_result[0] << endl;

    return 0;
}