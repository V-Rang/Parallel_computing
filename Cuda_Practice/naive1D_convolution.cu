/*
Naive 1D convolution: 
example array: [1,2,3,4,5]
example mask array: [1,2,3]

resulting convolution array: [8,14,20,26,14]
first element = a[0]*m[1] + a[2]*m[2];
last element = a[n-2]*m[0] + a[n-1]*m[1];

*/

#include<iostream>
#include<cuda.h>
#include<cuda_runtime.h>
#include<device_launch_parameters.h>
#include<assert.h>

using namespace std;

__global__ void convolution_1d(int *array, int *mask, int *result, int n, int m)
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    result[tid] = 0;
    int r = m/2;

    int start = tid - r;

    for(int j=0;j<m;j++)
    {
        if((start +j >= 0) && (start + j < n))
        {
            result[tid] += array[start+j] * mask[j];
        }
    }
}

void verify_result(int *array, int *mask, int *result, int n, int m)
{
    int r = m/2;

    for(int i=0;i<n;i++)
    {
        int temp = 0;
        int start = i - r;
        for(int j=0;j<m;j++)
        {
            if( (start + j >= 0) && (start + j < n) )
            {
                temp += array[start+j]*mask[j];
            }
        }
        assert(temp == result[i]);
    }
}

int main()
{
    int n = 1<<16;
    int m = 3;
    int *h_array, *h_mask, *h_result;
    int *d_array, *d_mask, *d_result;

    h_array = new int[n];
    h_mask = new int[m];
    h_result = new int[n];

    for(int i=0;i<n;i++) h_array[i] = rand()%100;
    for(int i=0;i<m;i++) h_mask[i] = rand()%10;

    cudaMalloc(&d_array,n*sizeof(int));
    cudaMalloc(&d_mask,m*sizeof(int));
    cudaMalloc(&d_result,n*sizeof(int));

    cudaMemcpy(d_array,h_array,n*sizeof(int),cudaMemcpyHostToDevice);
    cudaMemcpy(d_mask,h_mask,m*sizeof(int),cudaMemcpyHostToDevice);

    int ntb = 256;
    int nb = n/ntb;

    convolution_1d<<<nb,ntb>>>(d_array,d_mask,d_result, n, m);
    
    cudaMemcpy(h_result,d_result,n*sizeof(int),cudaMemcpyDeviceToHost);

    // for(int i=0;i<n;i++) cout << h_result[i] << " ";
    // cout << endl;

   verify_result(h_array,h_mask,h_result,n,m);
   printf("done\n");

    return 0;
}