/*
Modified 1d convolution to access elements beyond those in the shared array from the original array stored in DRAM.
*/

#include<iostream>
#include<cuda_runtime.h>
#include<device_launch_parameters.h>
#include<assert.h>

using namespace std;

#define MASK_LENGTH 3
__constant__ int mask[MASK_LENGTH];

// __global__ void conv(int *array, int *result, int n)
// {

//     int tid = blockDim.x * blockIdx.x + threadIdx.x;
//     extern __shared__ int s_array[];
//     s_array[threadIdx.x] = array[tid];
//     __syncthreads();
//     int offset = threadIdx.x + blockDim.x;
//     int g_offset = blockDim.x * blockIdx.x + offset;
//     int r = MASK_LENGTH/2;
//     int n_p = blockDim.x + 2*r;

//     if(offset < n_p)
//     {
//         s_array[offset] = array[g_offset];
//     }
//     __syncthreads();

//     int temp = 0;
//     for(int j=0;j<MASK_LENGTH;j++)
//     {
//         temp += s_array[threadIdx.x + j] * mask[j];
//     }
//     result[tid] = temp;
// }

__global__ void conv(int *array, int *result, int n)
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    extern __shared__ int s_array[];
    s_array[threadIdx.x] = array[tid];   
    __syncthreads();
    int temp = 0;
    for(int j=0;j<MASK_LENGTH;j++)
    {
        if( (threadIdx.x + j) >= blockDim.x )
        {
            temp += array[tid+j] * mask[j];
        }
        else
        {
            temp += s_array[threadIdx.x + j] * mask[j];
        }
    }
    result[tid] = temp;
}

void verify_result(int *array, int *mask, int *result, int n)
{
    // printf("calc values\n");
    for(int i=0;i<n;i++)
    {
        int temp = 0;
        for(int j=0;j<MASK_LENGTH;j++)
        {
            temp += array[i+j]*mask[j];
        }
        // cout << temp << endl;
        assert(temp == result[i]);
    }
}

int main()
{
    int *h_array, *h_mask, *h_result;
    int *d_array, *d_result;

    int n = 1<<13;
    int r = MASK_LENGTH/2;
    int n_p = n + 2*r; 

    h_array = new int[n_p];
    h_result = new int[n];
    h_mask = new int[MASK_LENGTH];

    cudaMalloc(&d_array,n_p*sizeof(int));
    cudaMalloc(&d_result,n*sizeof(int));
    h_mask[0] = 1;
    h_mask[1] = 2;
    h_mask[2] = 3;

    for(int i=0;i<n_p;i++)
    {
        if(  ( i<r) || (i+r == n_p) )
        {
            h_array[i] = 0;
        }
        else
        {
            h_array[i] = 1;
        }
    }

    // for(int i=0;i<n_p;i++) cout << h_array[i] << " ";
    // cout << endl;
    

    cudaMemcpy(d_array,h_array,n_p*sizeof(int),cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(mask,h_mask,MASK_LENGTH*sizeof(int));

    int ntb = 256;
    // int nb = n/ntb;
    // int nb = (n + ntb - 1)/ntb;
    int nb = n/ntb;
    conv<<<nb,ntb,ntb*sizeof(int)>>>(d_array,d_result,n);
    cudaMemcpy(h_result,d_result,n*sizeof(int),cudaMemcpyDeviceToHost);


    verify_result(h_array,h_mask,h_result,n);
    printf("done\n");

    delete[] h_array,h_mask,h_result;
    h_array = nullptr;
    h_result = nullptr;
    h_mask = nullptr;
    
    cudaFree(d_array);
    cudaFree(d_result);

    return 0;
}