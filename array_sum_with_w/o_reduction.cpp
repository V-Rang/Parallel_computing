#include<iostream>
#include <omp.h>

using namespace std;

int main()
{
    // int num_threds = omp_get_num_threads();
    // omp_set_num_threads(10);
    // int x;
    // #pragma omp parallel shared(x)
    // {
    //     x = 10;
    //     x += 1;
    //     printf("Hello from thread %d and value of x = %d\n",omp_get_thread_num(),x);
    // }

    omp_set_num_threads(16);
    //size and array to calculate sum of all elements
    int n = (int)1e4;
    int a[n] = {0};

    // initializing the array values
    for(int i=0;i<n;i++)
    {
        a[i] = i + 5;
    }

    //serial sum
    int ser_sum = 0;
    for(int i=0;i<n;i++)
    {
        ser_sum += a[i];
    }
    
    // parallel sum without using reduction
    int par_sum_wo_reduction = 0;
    int priv_sum;
    #pragma omp parallel private(priv_sum) shared(par_sum_wo_reduction)
    {
        priv_sum = 0;

        #pragma omp for
        for(int i=0;i<n;i++)
        {
            priv_sum += a[i];
        }   

        #pragma omp critical
        {
            par_sum_wo_reduction += priv_sum;
        }
    }
    
    
    // parallel sum using reduction
    int par_sum_w_reduction = 0;
    #pragma omp parallel for reduction (+:par_sum_w_reduction)
    for(int i=0;i<n;i++)
    {
        par_sum_w_reduction += a[i];
    }

    //results
    cout << "serial sum is "<< ser_sum << " " << "parallel sum with reduction "<< par_sum_w_reduction << " " << "parallel sum w/o reduction " << par_sum_wo_reduction << endl;
    
    return 0;
    
}