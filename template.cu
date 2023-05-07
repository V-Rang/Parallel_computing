#include<stdio.h>

//kernel routine
void __global__ my_kernel(float *x)
{
    int tid = threadIdx.x + blockDim.x*blockIdx.x;
    x[tid] = (float)tid;
}


int main(int argc, char **argv)
{
    float *h_x,*d_x;

  // number of blocks, and threads per block
    int nblocks=2;
    int nthreads = 8;
    int nsize = nblocks*nthreads;

  // allocate memory for array
    h_x = (float *)malloc(nsize*sizeof(float));
    cudaMalloc((void**)&d_x,nsize*sizeof(float));

  // kernel call
    my_kernel<<<nblocks,nthreads>>>(d_x);

  // copy results 
    cudaMemcpy(h_x,d_x,nsize*sizeof(float),cudaMemcpyDeviceToHost);

    for(int n=0;n<nsize;n++) printf("x = %f\n",h_x[n]);

  // free memory 
    cudaFree(d_x);
    free(h_x);

  // clean up
    cudaDeviceReset();

    return 0;
}