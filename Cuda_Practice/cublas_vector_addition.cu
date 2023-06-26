/*
saxpy - z = ax + y
*/

#include<iostream>
#include<cuda_runtime.h>
#include<assert.h>
#include<cublas_v2.h>
#include<stdlib.h>
#include<math.h>
#include<cuda.h>

using namespace std;

void init_vector(float *a, int s)
{
    for(int i=0;i<s;i++) a[i] = rand()%100;
}

void verify_result(float a, float *x, float *y, float *z, int n)
{
    for(int i=0;i<n;i++)
    {
        int temp =  a*x[i] + y[i];
        assert(temp == z[i]);
    }
}

int main()
{
    float *x,*y,*z;

    float *d_x, *d_y;

    const float a = 10; //factor
    int n = 1 << 10; // size of vector
    size_t size = n *sizeof(float);

    x = new float[n];
    y = new float[n];
    z = new float[n];

    init_vector(x,n);
    init_vector(y,n);
    
    cudaMalloc(&d_x,size);
    cudaMalloc(&d_y,size);

    cublasHandle_t handle;
    cublasCreate_v2(&handle);
    
    cublasSetVector(n,sizeof(float),x,1,d_x,1);
    cublasSetVector(n,sizeof(float),y,1,d_y,1);

    cublasSaxpy(handle,n,&a,d_x,1,d_y,1);
 
    cublasGetVector(n,sizeof(float),d_y,1,z,1);


    verify_result(a,x,y,z,n);

    printf("Successful\n");
    cublasDestroy(handle);

    cudaFree(d_x);
    cudaFree(d_y);

    delete[] x,y,z;

    return 0;

}
