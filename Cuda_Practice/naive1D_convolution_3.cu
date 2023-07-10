/*
1D Tiled Convolution:
If:
1. array: 1 1 1 1 1 1 
2. mask: 1 2 3

Expected convolution array is:
5 6 6 6 6 3

Concept of tiled 1d convolution:
array modified as: 0 1 1 1 1 1 1 0
calling 2 blocks of 3 threads each.
each block has a shared array: 
b0: 0 1 1 (0 1 2)
and b1: 1 1 1 (3 4 5)

Then, shared array for each block modified as 
b0: 0 1 1 1 1 (3 and 4)
b1: 1 1 1 1 0 (6 and 7)

*/


#include<iostream>
#include<cuda_runtime.h>
#include<device_launch_parameters.h>
#include<cuda.h>
#include<assert.h>

using namespace std;

#define MASK_LENGTH 3
__constant__ int mask[MASK_LENGTH];


// conv<<<nb,ntb>>>(d_array,d_result,n);
__global__ void conv(int *array,int *result, int n)
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;

    extern __shared__ int s_array[];

    int r = MASK_LENGTH/2;
    int d = 2*r;
    int n_padded = blockDim.x + d;
    int offset = threadIdx.x + blockDim.x;
    int g_offset = blockDim.x * blockIdx.x + offset;

    s_array[threadIdx.x] = array[tid];

    if(offset < n_padded)
    {
        s_array[offset] = array[g_offset];
    }
    __syncthreads();

    int temp = 0;

    for(int j=0;j<MASK_LENGTH;j++)
    {
        temp += s_array[threadIdx.x + j] * mask[j];
    }
    result[tid] = temp;
}

void verify_result(int *array, int *mask, int *result, int n)
{
    int temp;
    for(int i=0; i<n; i++)
    {
        temp = 0;
        for(int j=0;j<MASK_LENGTH;j++)
        {
            temp += array[i+j] * mask[j];
        }
        assert(temp == result[i]);
    }
}

int main()
{
    // int n = 1 << 20;
    int n = 6;
    int r = MASK_LENGTH/2;
    int n_p = n + r*2;
    
    int *h_array = new int[n_p];

    for(int i=0;i<n_p;i++)
    {
        if((i  < r) || (i >= (n+r)))
        {
            h_array[i] = 0;
        }
        else
        {
            // h_array[i] = rand()%100;
            h_array[i] = 1;
        }
    }

    int *h_mask = new int[MASK_LENGTH];

    // for(int i = 0; i< MASK_LENGTH; i++)
    // {
    //     h_mask[i] = rand()%10;
    // }

    h_mask[0] = 1;
    h_mask[1] = 2;
    h_mask[2] = 3;

    int *h_result = new int[n];

    int *d_array, *d_result;
    cudaMalloc(&d_array,n_p*sizeof(int));
    cudaMalloc(&d_result,n*sizeof(int));

    cudaMemcpy(d_array,h_array,n_p*sizeof(int),cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(mask,h_mask,MASK_LENGTH*sizeof(int));

    // int ntb = 256;
    // int nb = (n+ ntb - 1)/ntb;
    int ntb = 3;
    int nb  = n/ntb;

    size_t SHMEM  = (ntb + r * 2)/sizeof(int);
    conv<<<nb,ntb,SHMEM>>>(d_array,d_result,n);
    cudaMemcpy(h_result,d_result,n*sizeof(int),cudaMemcpyDeviceToHost);
    printf("array\n");
    for(int i=0;i<n_p;i++) cout << h_array[i] << " ";
    cout << endl;
    printf("mask\n");
    for(int i=0;i<MASK_LENGTH;i++) cout << h_mask[i] << " ";
    cout << endl;
    printf("result\n");
    for(int i=0;i<n;i++) cout << h_result[i] << " ";
    cout << endl;
    

    // verify_result(h_array,h_mask,h_result,n);

    printf("done\n");
    return 0;
}