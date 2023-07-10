/*
Improved 1d Convolution: use constant memory to declare "mask" array on device as it is used repeatedly; exploiting locality.
*/

#include<iostream>
#include<cuda_runtime.h>
#include<device_launch_parameters.h>

using namespace std;

#define MASK_LENGTH 3
__constant__ int mask[MASK_LENGTH];

__global__ void conv(int *array, int *result, int n)
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int temp = 0;
    int r = MASK_LENGTH/2;
    int start = tid - r;

    for(int j= 0;j<MASK_LENGTH;j++)
    {
        if( (start+j >= 0 ) && (start+j<n) )
        {
            temp += array[start+j] * mask[j];
        }
        result[tid] = temp;
    }
}

int main()
{
    int *h_array, *h_result,*h_mask;
    int *d_array, *d_result;
    int n = 5;

    h_array = new int[n]{1,1,1,1,1};
    h_mask = new int[MASK_LENGTH]{1,2,3};
    h_result = new int[n];

    cudaMalloc(&d_array,n*sizeof(int));
    cudaMalloc(&d_result,n*sizeof(int));

    int ntb = n;
    int nb = 1;

    cudaMemcpy(d_array,h_array,n*sizeof(int),cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(mask,h_mask,MASK_LENGTH*sizeof(int));

    conv<<<nb,ntb>>>(d_array,d_result,n);

    cudaMemcpy(h_result,d_result,n*sizeof(int),cudaMemcpyDeviceToHost);
    
    printf("array\n");
    for(int i=0;i<n;i++) cout << h_array[i] << " ";
    cout << endl;

    printf("mask\n");
    for(int i=0;i<MASK_LENGTH;i++) cout << h_mask[i] << " ";
    cout << endl;

    printf("result\n");
    for(int i=0;i<n;i++) cout << h_result[i] << " ";
    cout << endl;

    cudaFree(mask);
    cudaFree(d_array);
    cudaFree(d_result);

    delete[] h_array,h_result,h_mask;
    h_array = nullptr;
    h_result = nullptr;
    h_mask = nullptr;

    return 0;
}