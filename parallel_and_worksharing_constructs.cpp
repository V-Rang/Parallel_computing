/*example of orphaned work sharing construct:
Definition: work sharing constructs that are not enclosed within the lexical extent of a parallel region;
but are in the subroutine that is invoked directly or indirectly from inside a parallel region.
*/

#include<iostream>
#include<omp.h>

using namespace std;


int main()
{
    omp_set_num_threads(4);
    int tid;
    int i,j;
    #pragma omp parallel //create team of threads
    {
        tid = omp_get_thread_num();
        #pragma omp for //distribute the iterations to team of threads created above
        for(i = 0;i<4;i++)
        {
            #pragma omp parallel // spawn team of threads for each thread spawned in line 28
            {
                #pragma omp for // distribute the iterations to team of threads created above
                for(j=0;j<2;j++)
                {
                    printf("Hello from thread %d for i = %d j = %d\n",tid,i,j);
                }
            }
        }
    }

    return 0;
}