/*
Try for non square matrix:
A.T X B
where
A = [[1,4],
	 [2,5],
	 [3,6]]

B = [[1,2,3,4],
	 [5,6,7,8],
	 [9,10,11,12]]

C = A.T @ B
C = [[38,44,50,56],
 	[83,98,113,128]]

cublasSgemm documentation: 
m = number of rows of op(A) and C = 2.
n = number of cols of op(B) and C = 4.
k = number of cols of op(A) and rows of op(B) = 3.
lda = leading dimension of 2d array used to store the matrix A = 3.
lbd = leading dimension of 2d array used to store matrix B = 3.
ldc = leading dimension of 2d array used to store the matrix C = 2.
*/

#include<iostream>
#include<cuda.h>
#include<cuda_runtime.h>
#include<cublas.h>
#include<assert.h>

using namespace std;

__global__ void init(int *test, int n)
{
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	if(tid < n) test[tid] = 1;
}

void verify_results(float *x, float *y, float *z, int n)
{
	for(int i=0;i<n;i++)
	{
		for(int j=0;j<n;j++)
		{
			float temp = 0;
			for(int k=0;k<n;k++)
			{
				temp += x[k*n + j] * y[i*n + k];
			}
			assert(fabs(temp - z[i*n+j]) < 0.001);

		}
	}
}

int main()
{
	// int n = 2;
	// size_t size = n*n*sizeof(int);

	float *h_x,*h_y,*h_z;
	float *d_x,*d_y,*d_z;

	h_x = new float[6]{1,2,3,4,5,6};
	h_y = new float[12]{1,5,9,2,6,10,3,7,11,4,8,12};
	h_z = new float[8];

	cudaMalloc(&d_x,6*sizeof(float));
	cudaMalloc(&d_y,12*sizeof(float));
	cudaMalloc(&d_z,8*sizeof(float));
	
	cudaMemcpy(d_x,h_x,6*sizeof(float),cudaMemcpyHostToDevice);
	cudaMemcpy(d_y,h_y,12*sizeof(float),cudaMemcpyHostToDevice);
	
	cublasSgemm('T','N',2,4,3,1.0,d_x,3,d_y,3,0.0,d_z,2);

	cudaMemcpy(h_z,d_z,8*sizeof(float),cudaMemcpyDeviceToHost);

	// for(int i=0;i<8;i++) cout << h_z[i] << " ";
	// cout << endl;

	int m = 2; //no of rows in C
	int n = 4; //no of cols in C
	
	printf("In matrix form:\n");
	for(int i=0;i<m;i++) //row
	{
		for(int j=0;j<n;j++) //col
		{
			cout << h_z[j*m+i] <<  " "; //h_z receives values in col major order
		}
		cout << endl;
	}

	//transpose of h_c
	printf("Transpose in matrix form:\n");
	for(int i=0;i<n;i++)
	{
		for(int j=0;j<m;j++)
		{
			cout << h_z[i*m +j ]  << " ";
		}
		cout << endl;
	}

	// verify_results(h_x,h_y,h_z,n);
	// printf("Successful\n");


	return 0;
}
