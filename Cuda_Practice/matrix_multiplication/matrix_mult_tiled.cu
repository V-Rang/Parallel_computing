#include<iostream>
#include<cuda.h>
#include<cuda_runtime.h>
#include<vector>
#include<algorithm>
#include<assert.h>

using namespace std;

const int num_threads = 32; //blockDim.x = blockDim.y
const int n = 1 << 10; // order of square matrices

__global__ void matrixmult(int *a, int *b, int *c)
{
    __shared__ int shared_a[num_threads][num_threads]; //shared elements are shared by all threads of a block
    __shared__ int shared_b[num_threads][num_threads];

    // int blockwidth = n/num_threads;

    int by = blockIdx.y;
    int bx = blockIdx.x;

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = blockDim.y * by + ty;
    int col = blockDim.x * bx + tx;
    
    int temp = 0; //each thread (x,y) calculates a given value in c.

    for(int i=0;i<n;i+=blockDim.x)
    {
        shared_a[ty][tx] = a[row*n + i + tx];
        shared_b[ty][tx] = b[(i + ty)*n + col];
        __syncthreads(); //s_a and s_b must be finished creating before moving to next step
 
        for(int j=0;j<num_threads;j++)
        {
            temp += shared_a[ty][j]*shared_b[j][tx];        
        }
        __syncthreads(); //temp of current s_a and s_b block must be finished before moving to creating that of next step, i.e. s_a and s_b must not be polluted by values of the next iteration before this temp is finsihed creating.
    }
    c[row*n + col] = temp;
}


void verify_result(vector<int>&a, vector<int>&b, vector<int>&c, int n)
{
    for(int i=0;i<n;i++) // each row
    {
        for(int j=0;j<n;j++) // each col
        {
            int temp = 0;
            for(int k=0;k<n;k++)
            {
                temp += a[i*n + k]*b[k*n + j];
            }
            assert(temp == c[i*n + j]);
        }
    }
}   

int main()
{
    size_t size = n * n * sizeof(int);
    vector<int>h_a(n*n);
    vector<int>h_b(n*n);
    vector<int>h_c(n*n);

    generate(h_a.begin(), h_a.end() , [](){return rand()%100;});
    generate(h_b.begin(), h_b.end() , [](){return rand()%100;});

    int *d_a, *d_b, *d_c;

    cudaMalloc(&d_a,size);
    cudaMalloc(&d_b,size);
    cudaMalloc(&d_c,size);

    // int num_threads = 32;
    int num_blocks = n/num_threads;

    dim3 threads(num_threads,num_threads);
    dim3 blocks(num_blocks,num_blocks);

    cudaMemcpy(d_a,h_a.data(),size,cudaMemcpyHostToDevice);
    cudaMemcpy(d_b,h_b.data(),size,cudaMemcpyHostToDevice);

    matrixmult<<<blocks,threads>>>(d_a,d_b,d_c);

    cudaMemcpy(h_c.data(),d_c,size,cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    cudaFree(d_b);  
    cudaFree(d_c);

    verify_result(h_a,h_b,h_c,n);

    printf("successful");
    
}
