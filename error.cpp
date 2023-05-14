/*
Not sure how this code works.
*/


#include<iostream>
#include<omp.h>

using namespace std;


int main()
{
    omp_set_num_threads(5);
    int tid;
    int i,j;
    #pragma omp parallel private(tid)
    {
        tid = omp_get_thread_num();
        
        if(tid %2 == 0)
        {
            #pragma omp for 
            for(i=0;i<5;i++)
            {
                printf("Hello from even numbered thread %d for i = %d\n",tid,i);
            }
        }
        else
        {
            #pragma omp for private(j)
            for(j=0;j<5;j++)
            {
                printf("Hello from odd numbered thread %d for j = %d\n",tid,j);
            }

        }
    }

    return 0;
}