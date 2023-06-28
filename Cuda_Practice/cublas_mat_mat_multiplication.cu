#include<iostream>
#include<cublas.h>
#include<assert.h>
#include<curand.h>

using namespace std;

void verify_results(float *a, float *b, float *c, int n)
{
    float temp;
    for(int i=0;i<n;i++)
    {
        for(int j=0;j<n;j++)
        {
            temp = 0;
            for(int k = 0;k<n;k++)
            {
                temp += a[k*n + j] * b[i*n + k];
            }
            assert(fabs(temp - c[i*n+j]) < 1e-3);
        }
    }
}

// void verify_solution(float *a, float *b, float *c, int n)
// {
//     float temp;
//     float epsilon = 0.001;
//     for(int i=0;i<n;i++)
//     {
//         for(int j=0;j<n;j++)
//         {
//             temp = 0;
//             for(int k=0;k<n;k++)
//             {
//                 temp += a[k*n + i]*b[j*n + k];
//             }
//             assert(fabs(c[j*n + i] - temp) < epsilon);
//         }
//     }
// }

int main()
{
    float *h_a, *h_b, *h_c;
    float *d_a, *d_b, *d_c;

    // int n = 2;
    int n = 1 << 10;
    size_t size = n*n*sizeof(float);

    h_a = new float[n*n];
    h_b = new float[n*n];
    h_c = new float[n*n];

    cudaMalloc(&d_a,size);
    cudaMalloc(&d_b,size);
    cudaMalloc(&d_c,size);

    curandGenerator_t prng;
    curandCreateGenerator(&prng,CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(prng,(unsigned long long)clock());

    curandGenerateUniform(prng,d_a,size);
    curandGenerateUniform(prng,d_b,size);
     
    cublasHandle_t handle;
    cublasCreate_v2(&handle);

    float alpha = 1;
    float beta = 0;

    cublasSgemm('N','N',n,n,n,alpha,d_a,n,d_b,n,beta,d_c,n);

    cudaMemcpy(h_c,d_c,size,cudaMemcpyDeviceToHost);
    cudaMemcpy(h_a,d_a,size,cudaMemcpyDeviceToHost);
    cudaMemcpy(h_b,d_b,size,cudaMemcpyDeviceToHost);

    //C++ verification code
    verify_results(h_a,h_b,h_c,n);
    printf("Successful\n");


    return 0;
}
