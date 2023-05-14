/*example of orphaned work sharing construct:
Definition: work sharing constructs that are not enclosed within the lexical extent of a parallel region;
but are in the subroutine that is invoked directly or indirectly from inside a parallel region.



*/

#include<iostream>
#include<omp.h>

using namespace std;

void hello(int id)
{
    #pragma omp for 
    for(int i=0;i<4;i++)
    {
        printf("Hello from thread %d\n",id);
    }

}


int main()
{
    omp_set_num_threads(4);
    int tid;
    #pragma omp parallel private(tid) //if comment this and run program as is, similar to running entire program serially
    {
        tid = omp_get_thread_num();
        hello(tid);
    }  

    return 0;
}