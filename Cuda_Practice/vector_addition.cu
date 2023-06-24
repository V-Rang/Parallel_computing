#include<cuda.h>
#include<cuda_runtime.h>
#include<iostream>

using namespace std;

__global__ void vectorAdd(float *x, float *y, float *z, int n)
{
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	if(tid < n) z[tid] = x[tid] + y[tid];
}

int main()
{
	float *a, *b, *c;
	float *d_a,*d_b,*d_c;

	int n = 5;

	a = new float[5]{1,2,3,4,5};
	b = new float[5]{11,12,13,14,15};

	c = new float[5];

	int size = n * sizeof(float);

	cudaMalloc(&d_a,size);
	cudaMalloc(&d_b,size);
	cudaMalloc(&d_c,size);

	cudaMemcpy(d_a,a,size,cudaMemcpyHostToDevice);
	cudaMemcpy(d_b,b,size,cudaMemcpyHostToDevice);
	cudaMemcpy(d_c,c,size,cudaMemcpyHostToDevice);
	
	vectorAdd<<<1,1024>>>(d_a,d_b,d_c,n);

	cudaMemcpy(c,d_c,size,cudaMemcpyDeviceToHost);

	for(int i=0;i<n;i++) std::cout << c[i] << " ";
	std::cout << endl;
	


	return 0;
}
