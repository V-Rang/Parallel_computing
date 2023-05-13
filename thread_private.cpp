/*working of threadprivate. Adapted from https://hpc-tutorials.llnl.gov/openmp/threadprivate_directive/

Difference between private and threadprivate:
1. Storage(S): most likely placed in stack | most likely in heap or thread local storage -> i.e. global memory local to thread.
2. Duration (D): Lifetime of variables is duration of data scoping clause | Threadprivate variable persists across regions.
3. Copy (C): Every thread including master thread makes a private copy of original variable |
Master thread uses original variable, all other threads make a private copy of the original variable. 

*/

#include<iostream>
#include<omp.h>

int a,b,tid;
float x;

#pragma omp threadprivate(a,x)


int main()
{
    omp_set_num_threads(4);
    b = 12;
    #pragma omp parallel private(tid,b)
    {
        tid = omp_get_thread_num();
        a = tid;
        b = tid;
        x = 1.1*tid + 1.0;
        printf("Thread %d: a,b,x = %d %d %f\n",tid,a,b,x);
    }
    printf("***************\n");
    printf("Master thread here\n");
    printf("***************\n");

    #pragma omp parallel private(tid)
    {
        tid = omp_get_thread_num();
        printf("Thread %d: a,b,x = %d %d %f\n",tid,a,b,x);
    }
}
using namespace std;
