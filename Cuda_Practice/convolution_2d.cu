/*
2D Convolution
Working: 
Consider example: 
array:
1 2 3
4 5 6
7 8 9
and 
mask:
1 2 
3 4

Then: 
result is a 9 length array where:
r[0] is calculated as:
1 2 
3 4 
(the mask)
being layed over the array such that 
4 (from the mask) is above 1 (from the array).

for r1 ,r2 move the mask left.

for r3 move mask down i.e,
the mask is layed over the array such that
2 (from the mask) is above 1 (from the array) and 
4 (from the mask) is above 4 (from the array)

Similarly proceed for the other values in the result array.
*/

#include<iostream>
#include<cuda_runtime.h>
#include<device_launch_parameters.h>
#include<assert.h>

using namespace std;

#define MASK_DIM 7
#define MASK_OFFSET (MASK_DIM/2)

__constant__ int mask[7*7];

__global__ void conv_2d(int *matrix, int *result, int N)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    int start_r = row - MASK_OFFSET;
    int start_c = col - MASK_OFFSET;

    int temp = 0;

    for(int i=0; i<MASK_DIM; i++)
    {
        for(int j=0;j<MASK_DIM;j++)
        {
            if((start_r+i) >= 0 && (start_r + i) < N)
            {
                if((start_c + j) >= 0 &&  (start_c + j) < N)
                {
                    temp += matrix[(start_r + i)*N + (start_c+j)] * mask[i*MASK_DIM + j];
                }
            }
        }
    }
    result[row*N + col] = temp;
}

void init_matrix(int *m, int n)
{
    for(int i=0; i< n; i++)
    {
        for(int j=0;j<n; j++)
        {
            m[n*i+j] = rand()%100;
        }
    }
}

void verify_result(int *m, int *mask, int *result, int N)
{
    int temp;
    int offset_r;
    int offset_c;

    for(int i=0;i<N;i++)
    {
        for(int j=0;j<N;j++)
        {
            temp = 0;
            for(int k=0;k<MASK_DIM;k++)
            {
                offset_r = i - MASK_OFFSET + k;
                for(int l=0; l < MASK_DIM; l++)
                {
                    offset_c = j - MASK_OFFSET + l;
                    if(offset_r >= 0 && offset_r < N)
                    {
                        if(offset_c >=0 && offset_c < N)
                        {
                            // printf("i = %d, j = %d, i*N+j= %d, index of matrix = %d, index of mask = %d\n",i,j,i*N+j,offset_r*N+offset_c, k*MASK_DIM+l);
                            temp += m[offset_r*N + offset_c] * mask[k*MASK_DIM + l];
                        }
                    }
                }
            }
            assert(result[i*N+j] == temp);
        }
    }
}

int main()
{
    int N = 1 << 10;
    size_t bytes_n = N*N*sizeof(int);

    int *matrix = new int[N*N];
    int *result = new int[N*N];

    init_matrix(matrix,N);
    
    size_t bytes_m = MASK_DIM * MASK_DIM * sizeof(int);

    int *h_mask =  new int[MASK_DIM * MASK_DIM];
    init_matrix(h_mask,MASK_DIM);
    // h_mask[0] = 1;
    // h_mask[1] = 2;
    // h_mask[2] = 3;
    // h_mask[3] = 4;
    
    int *d_matrix;
    int *d_result;

    cudaMalloc(&d_matrix,bytes_n);
    cudaMalloc(&d_result,bytes_n);

    cudaMemcpy(d_matrix,matrix,bytes_n,cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(mask,h_mask,bytes_m);

    // int ntb = 16;
    // int nb = (N+ntb-1)/ntb;
    int ntb = 16;
    int nb = (N + ntb - 1)/ntb;

    dim3 block_dim(ntb,ntb);
    dim3 grid_dim(nb,nb);

    conv_2d<<<grid_dim,block_dim>>>(d_matrix,d_result,N);

    cudaMemcpy(result,d_result,bytes_n,cudaMemcpyDeviceToHost);
    
    verify_result(matrix,h_mask,result,N);
    // for(int i=0;i<N*N;i++)
    // {
    //     cout << result[i] << endl;
    // }
    
    printf("done\n");

    delete[] matrix,result,h_mask;
    matrix = nullptr;
    result = nullptr;
    h_mask = nullptr;

    cudaFree(d_matrix);
    cudaFree(d_result);
    cudaFree(mask);


    return 0;
}