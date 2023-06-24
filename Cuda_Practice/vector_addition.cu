#include<cuda.h>
#include<cuda_runtime.h>
#include<iostream>
#include<assert.h>

using namespace std;

__global__ void vectorAdd(float *a, float *b, float*c, int n)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if(tid < n) c[tid] = a[tid] + b[tid];
}


void verify_result(float *&a, float *&b, float *c, int n)
{
	for(int i=0;i<n;i++)
	{
		// printf("iteration = %d\n",i);
		assert(c[i] == a[i] + b[i] );
	}
}

int main()
{
	int n = 1 << 16;
	// int n = 10;
	size_t size = n * sizeof(float);

	float *h_x,*h_y,*h_z;

	float *d_x,*d_y,*d_z;

	h_x = new float[n];
	h_y = new float[n];
	h_z = new float[n];

	cudaMalloc(&d_x,size);
	cudaMalloc(&d_y,size);
	cudaMalloc(&d_z,size);

	for(int i=0;i<n;i++)
	{
		h_x[i] = rand()%100;
		h_y[i] = rand()%100; 
	}


	cudaMemcpy(d_x,h_x,size,cudaMemcpyHostToDevice);
	cudaMemcpy(d_y,h_y,size,cudaMemcpyHostToDevice);
	cudaMemcpy(d_z,h_z,size,cudaMemcpyHostToDevice);

	int num_threads = 256;
	int num_blocks = n/num_threads;

	vectorAdd<<<num_blocks,num_threads>>>(d_x,d_y,d_z,n);
	// vectorAdd<<<1,1024>>>(d_x,d_y,d_z,n); //this gives error with 1 << 16 elements.

	cudaMemcpy(h_z,d_z,size,cudaMemcpyDeviceToHost);

	// for(int i=0;i<n;i++)
	// {
	// 	cout << h_x[i] << " " << h_y[i] << " " << h_z[i] << endl;
	// }

	verify_result(h_x,h_y,h_z,n);
	
	
	cudaFree(d_x);
	cudaFree(d_y);
	cudaFree(d_z);

	return 0;
}



