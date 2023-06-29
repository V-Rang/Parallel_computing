#include<cuda.h>
#include<cuda_runtime.h>
#include<iostream>

using namespace std;

__global__ void check()
{

	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	printf("Hello from thread %d\n",tid);
}

int main()
{
	int num_blocks = 2;
	int num_threads = 8;
	check<<<num_blocks,num_threads>>>();	
	cudaDeviceSynchronize(); //have to add this to ensure all threads are done executing the kernel before control returned to host

	return 0;
}
