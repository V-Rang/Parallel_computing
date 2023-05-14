/*example of orphaned work sharing construct:
Definition: work sharing constructs that are not enclosed within the lexical extent of a parallel region;
but are in the subroutine that is invoked directly or indirectly from inside a parallel region.



*/

#include<iostream>
#include<omp.h>

using namespace std;

void hello(int id)
{
    /*
    if use #pramga omp parallel for: parallel from main invokes team of threads -> for each of the threads -> a team
    is spawned again and the iterations of i are shared. This is done for each thread created from the parallel call
    from main. Therefore total outputs = number of threads (from main) X number of iterations of i


    if use #pragma omp for: parallel from main invokes team of threads -> each of threads take a portion of the 
    iterations of i-> i.e. total number of outputs = number of iterations of i.
    
    */
    // #pragma omp parallel for
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