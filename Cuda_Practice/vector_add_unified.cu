#include<iostream>
#include<cuda.h>
#include<cuda_runtime.h>
#include<cassert>
// using namespace std;

using std::cout;
using std::endl;

__global__ void vectorAdd(int *a, int *b, int *c, int n)
{
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	if(tid < n) c[tid] = a[tid] + b[tid];
}

void verify_result(int *a, int *b, int *c, int n)
{
	for(int i=0;i<n;i++) assert(c[i] == a[i] + b[i]);
}

int main()
{

	int id = cudaGetDevice(&id);
	int n = 1 << 16;
	size_t size = n * sizeof(int);
	
	int *a = new int[n];
	int *b = new int[n];
	int *c = new int[n];

	cudaMallocManaged(&a,size);
	cudaMallocManaged(&b,size);
	cudaMallocManaged(&c,size);

	for(int i=0;i<n;i++)
	{
		a[i] = rand()%100;
		b[i] = rand()%100;
	}

	// int num_blocks = 1 << 10;
	// int num_grids = (n + num_blocks - 1)/num_blocks;

	int num_grids = 256;
	int num_blocks = 256;

	//we need all data on GPU. it will start taking data "page-by-page" from the CPU once it 
	//encounters the below function vectorAdd. To speed things up, we can ask it to start
	//prefetching data in the background:

	cudaMemPrefetchAsync(a,size,id);
	cudaMemPrefetchAsync(b,size,id);
	vectorAdd<<<num_grids,num_blocks>>>(a,b,c,n);

	//because kernel calls are not synchronous from CPU side. Therefore, need to
	//wait for above computation to be complete before verifying results.
	//didnt have to do this for vector_addition b/c cudaMemcpy is synchronous.

	cudaDeviceSynchronize(); //wait for compute device to finish.
	cudaMemPrefetchAsync(c,size,cudaCpuDeviceId);

	verify_result(a,b,c,n);

	cudaFree(a);
	cudaFree(b);
	cudaFree(c);


	return 0;
}

