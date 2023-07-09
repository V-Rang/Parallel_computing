/*
Summing 2^13 = 8192 array of ones using cooperative groups. 
Idea: 
Motivation for cooperative groups: 
safer coding to prevent deadlocks. If have code like "if(threadIdx.x < .. )" then call a routine and in that routine you have __syncthreads().
This will cause a deadlock as some threads of a block never entered that routine. 
To mitigate this, you call a routine using groups of thread within that thread block and then call group.sync() for all the threads of that group 
that you know entered that routine. 

Working of code:
1. Initialize array of 2^16 elements -> all set to 1.
2. Then all the threads calculate sum of 4 elements at once:
block 0:
threads: 0 - 255.
0 calculates (0-3) elements of the array.
1 calculates (4-7) elements of the array.
...
255 calculates (1020-1023) elements of the array.

We call 256 blocks, but block 7 is the last one that has non-zero values.

block 7:
threads: 1792 - 2047
...
2047 calculates (8188-8191) elements of the array.


Then we call block_sum and thread 0 of all the blocks contains the sum of the elements that each thread of the block contains.

We finally call atomicAdd on reduction of the value contained by thread 0 of all the blocks.

We can similarly do for 2^16 = 65536 elements using 64 blocks, i.e., 
block 63:
threads: 16128 - 16383
16128 calculates (64512 - 64515) elements of the array.
...
16383 calculates (65532 - 65535) elements of the array.

*/


#include<iostream>
#include<cuda_runtime.h>
#include<device_launch_parameters.h>
#include<cuda.h>
#include<cooperative_groups.h>

using namespace std;
using namespace cooperative_groups;

void initialize_vect(int *data, int n)
{
    for(int i=0;i<n;i++) data[i] = 1;
}


__device__ int thread_sum(int *input, int n)
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int my_sum = 0;
    for(int i=tid; i< n/4; i += blockDim.x * gridDim.x)
    {
        int4 in = ((int4*)input)[i];
        my_sum += in.x + in.y + in.z + in.w;
    }
    return my_sum;
}

__device__ int block_sum(thread_block g, int my_sum, int*temp)
{
    int lane = g.thread_rank();
    
    for(int i=g.size()/2 ; i > 0; i /= 2)
    {
        temp[lane] = my_sum;
        g.sync();
        if(lane < i)
        {
            my_sum += temp[lane + i];
        }
        g.sync();
    }
    return my_sum;
}

__global__ void sum_reduction(int *sum, int *input, int n)
{
    // int tid = blockDim.x * blockIdx.x + threadIdx.x;
    extern __shared__ int temp[];
    int my_sum = thread_sum(input,n);
    // printf("Thread %d has the sum  = %d\n",tid,my_sum);
    auto g = this_thread_block();
    int bsum = block_sum(g,my_sum,temp);

    // if(tid == 256) printf("Value in thread %d = %d\n",tid,bsum);
    if(g.thread_rank() == 0)
    {
        atomicAdd(sum,bsum);
    }
}

int main()
{
    int *data,*sum;
    // int n = 1<<13;
    int n = 1<<16;
    cudaMallocManaged(&sum,sizeof(int));
    cudaMallocManaged(&data,n*sizeof(int));
    initialize_vect(data,n);
    int ntb = 256;
    // int nb = n/ntb;
    // int nb = 7;
    int nb = 64;
    sum_reduction<<<nb,ntb,256*sizeof(int)>>>(sum,data,n);
    cudaDeviceSynchronize();
    cout << sum[0] << endl;

    return 0;
}