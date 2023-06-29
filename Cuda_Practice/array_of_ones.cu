#include<iostream>
#include<cuda.h>
#include<cuda_runtime.h>

using namespace std;

__global__ void init(int *test, int n)
{
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	if(tid < n) test[tid] = 1;
}


int main()
{
	int n = 16;
	int *h_x = new int[n];
	for(int i=0;i<n;i++) h_x[i] = 0;

	int *d_x;

	cudaMalloc(&d_x,n*sizeof(int));
	cudaMemcpy(d_x,h_x,n*sizeof(int),cudaMemcpyHostToDevice);
	for(int i=0;i<n;i++) cout << h_x[i] <<  " ";
	cout << endl;


	init<<<1,32>>>(d_x,n);

	cudaMemcpy(h_x,d_x,n*sizeof(int),cudaMemcpyDeviceToHost);

	for(int i=0;i<n;i++) cout << h_x[i] <<  " ";
	cout << endl;
	return 0;
	
}
