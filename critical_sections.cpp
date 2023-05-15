#include<iostream>
#include<omp.h>
#include<climits>

using namespace std;


int main()
{
    omp_set_num_threads(5);
    int tid;
    int i,j;
    
    int a[5] = {1,13,45,-56,-32};
    int cur_max = -INT_MAX;

    //Codes to find maximum element in the array using critical section

    //M-1 inefficient critical - effectively seralized
    // #pragma omp parallel for
    // for(i=0;i<5;i++)
    // {
    //     #pragma omp critical
    //     {
    //         if(a[i]>cur_max)
    //         {
    //             cur_max = a[i];
    //         }
    //     }
    // }


    //M-2 improved critcal section
    /*
    Reason: no need to have threads queuing up to enter critical section if cur_max is already greater than a[i].
    */
    #pragma omp parallel for
    for(i=0;i<5;i++)
    {
        if(a[i]>cur_max)
        {
            #pragma omp critical
            {
                if(a[i]>cur_max) //have to again compare cur_max and a[i] because the value of cur_max may have changed in the meantime
                {
                    cur_max = a[i];
                }
            }
        }
    }


    cout << cur_max << endl;


    return 0;
}