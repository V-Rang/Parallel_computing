//adding two vectors

#include<stdio.h>



void __global__ vec_add(float*x,float*y)
{
    int tid = threadIdx.x + blockDim.x*blockIdx.x;

    y[tid] += x[tid];
}


int main(int argc, char **argv)
{
    float *h_x,*h_y,*d_x,*d_y;
    int nblocks = 2;
    int nthreads = 4;
    int nsize = nblocks*nthreads;

    h_x = (float*)malloc(nsize*sizeof(float));
    h_y = (float*)malloc(nsize*sizeof(float));
 
    for(int i=0;i<nsize;i++)
    {
        h_x[i] = i+5;
    }

    for(int i=0;i<nsize;i++)
    {
        h_y[i] = i+2;
    }

    for(int i=0;i<nsize;i++) printf("h_x[i] = %f\n",h_x[i]);
    for(int i=0;i<nsize;i++) printf("h_y[i] = %f\n",h_y[i]);

    // for(int i=0;i<nsize;i++) printf("h_x[i] = %f\n",h_x[i]);
    cudaMalloc((void **)&d_x,nsize*sizeof(float));
    cudaMalloc((void **)&d_y,nsize*sizeof(float));

    cudaMemcpy(d_x,h_x,nsize*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(d_y,h_y,nsize*sizeof(float),cudaMemcpyHostToDevice);
 
    vec_add<<<nblocks,nthreads>>>(d_x,d_y);

    cudaMemcpy(h_y,d_y,nsize*sizeof(float),cudaMemcpyDeviceToHost);

    for(int i=0;i<nsize;i++) printf("ans[i] = %f\n",h_y[i]);

    free(h_x);
    free(h_y);
    cudaFree(d_x);
    cudaFree(d_y);

    cudaDeviceReset();

    return 0;
}