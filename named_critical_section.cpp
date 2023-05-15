/*
Named critical section. By default, only one critical section can be executed at once in a progam.
Naming critical sections allows simultaneous execution of multiple critcal sections at once. Sections with the same
name have dependency and so, are not executed simultaneously.
If unnnamed critical section is a global lock, named critical sections are like lock variables.
*/

#include<iostream>
#include<omp.h>
#include<climits>

using namespace std;


int main()
{
    omp_set_num_threads(5);
    int tid;
    int i,j;
    
    int cur_max = -INT_MAX;
    int cur_min = INT_MAX;
    int a[5] = {-3,45,114,-56,23};

    #pragma omp parallel for
    for(i=0;i<5;i++)
    {
        if(a[i]>cur_max)
        {
            #pragma omp critical (MAXLOCK)
            {
                if(a[i]>cur_max)
                {
                    cur_max = a[i];
                }
            }
        }
        if(a[i]<cur_min)
        {
            #pragma omp critical (MINLOCK)
            {
                if(a[i]<cur_min)
                {
                    cur_min = a[i];
                }
            }
        }
    }

    printf("The maximum value in the array is %d.\nThe minimum value in the array is %d.\n",cur_max,cur_min);

    return 0;
}