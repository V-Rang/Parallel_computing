#include<iostream>
#include<cuda.h>
#include<cuda_runtime.h>
#include<algorithm>
#include<vector>
#include<assert.h>

using namespace std;

__global__ void matrixmult(int *a, int *b, int *c, int n)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    c[row*n + col] = 0;
    for(int k=0;k<n;k++) c[row*n + col] += a[row*n + k] * b[k*n + col];
}

void verify_result(vector<int>&a, vector<int>&b, vector<int>&c, int n)
{
    for(int i=0;i<n;i++) // row
    {
        for(int j=0;j<n;j++) //col
        {
            int temp = 0;
            for(int k=0;k<n;k++)
            {
                temp += a[i*n + k] * b[k*n + j];
            }
            assert(temp == c[i*n + j]);
        }
    }
}

int main()
{
    int n = 1 << 10; //has to be divisible by num_threads, i.e., num_threads X int (i.e. num_blocks) = n.
    /*
    Reason for above: let n = 20, i.e., 20 X 20 matrix, if num_threads along x and y each are set to 7, then
    the number of blocks = 20/7 = 2.
    So, in the GPU call: row: 0 ... 6, 7 ... 13, and col: 0 ... 6, 7 ... 13. but we had to solve for
    row = 0...19 and col = 0...19. So you end up with h_c containing the sub 14 X 14 matrix of the 
    actual 20 X 20 matrix.
    */
    // int n = 20;
    

    vector<int>h_a(n*n);
    vector<int>h_b(n*n);
    vector<int>h_c(n*n);
    
    int *d_a, *d_b, *d_c;

    size_t size = n*n*sizeof(int);

    cudaMalloc(&d_a,size);
    cudaMalloc(&d_b,size);
    cudaMalloc(&d_c,size);

    generate(h_a.begin(), h_a.end(), [](){return rand()%100;});
    generate(h_b.begin(), h_b.end(), [](){return rand()%100;});
    

    cudaMemcpy(d_a,h_a.data(),size,cudaMemcpyHostToDevice);
    cudaMemcpy(d_b,h_b.data(),size,cudaMemcpyHostToDevice);
    
    int num_threads = 32;
    // int num_threads = 7;
    int num_blocks = n/num_threads;

    dim3 threads(num_threads,num_threads);
    dim3 blocks(num_blocks,num_blocks);

    matrixmult<<<blocks,threads>>>(d_a,d_b,d_c,n);
    
    cudaMemcpy(h_c.data(),d_c,size,cudaMemcpyDeviceToHost);

    verify_result(h_a,h_b,h_c,n);

    printf("Success!\n");

    return 0;

    
}
