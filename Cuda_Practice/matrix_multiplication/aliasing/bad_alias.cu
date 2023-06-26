/*
(transposing B and then traversing the elements of its rows, i.e. each thread accesses a unique row of Bt, each of which are displaced far 
from each other in memory. This results in bad memory usage as data for threads cannot be found with single call to DRAM that will load the 
values needed by the threads in the cache.)
*/

#include<iostream>
#include<cuda.h>
#include<cuda_runtime.h>
#include<vector>
#include<algorithm>
#include<assert.h>

using namespace std;

__global__ void matrix_mult(int *a, int *b, int *c, int n)
{
    int row = blockIdx.y*blockDim.y + threadIdx.y;
    int col = blockIdx.x*blockDim.x + threadIdx.x;

    c[row*n + col] = 0;

    for(int i=0;i<n;i++) c[row*n+col] += a[row*n + i]*b[col*n + i];
}

void verify_result(vector<int>&a, vector<int>&b, vector<int>&c, int n)
{
    for(int i=0;i<n;i++) //for each row
    {
        for(int j=0;j<n;j++) //for each col
        {
            int temp = 0;
            for(int k=0;k<n;k++)
            {
                temp += a[i*n + k]*b[k*n + j];
            }
            assert(temp == c[i*n+j]);
        }
    }
}

void transposer(vector<int>&a, vector<int>&a_t,int n)
{
    for(int i=0;i<n;i++)
    {
        for(int j=0;j<n;j++)
        {
            a_t[j*n+i] = a[i*n+j];
        }
    }
}

int main()
{
    int n = 1 << 10;
    int num_threads = 32;
    // int n = 4;
    // int num_threads = 2;
    int num_blocks = n/num_threads;
    size_t size = n * n * sizeof(int);

    dim3 threads(num_threads,num_threads);
    dim3 blocks(num_blocks,num_blocks);

    vector<int>h_a(n*n), h_b(n*n), h_c(n*n);

    vector<int>h_b_t(n*n);

    int *d_a, *d_b, *d_c;

    cudaMalloc(&d_a,size);
    cudaMalloc(&d_b,size);
    cudaMalloc(&d_c,size);  

    generate(h_a.begin(), h_a.end(), [](){return rand()%100;});
    generate(h_b.begin(), h_b.end(), [](){return rand()%100;});

    transposer(h_b,h_b_t,n);


    cudaMemcpy(d_a,h_a.data(),size,cudaMemcpyHostToDevice);
    cudaMemcpy(d_b,h_b_t.data(),size,cudaMemcpyHostToDevice);
    
    matrix_mult<<<blocks,threads>>>(d_a,d_b,d_c,n);

    cudaMemcpy(h_c.data(),d_c,size,cudaMemcpyDeviceToHost);

    verify_result(h_a,h_b,h_c,n);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    

    printf("Successful\n");

    return 0;
}
