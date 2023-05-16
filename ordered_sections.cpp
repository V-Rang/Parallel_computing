/*
example of outputing values in an array in order using ordered sections - a type of synchronization event.
Types of synchronization events:
    1. barrier directive
    2. ordered sections
    3. master directive
*/


#include<iostream>
#include<omp.h>
#include<climits>
#include<bitset>
#include<random> 


using namespace std;


int main()
{
    omp_set_num_threads(5);
    int tid;
    int i,j;

    int a[5] = {0};

    #pragma omp parallel for
    for(i=0;i<5;i++)
    {
        a[i] += i;
    }

    printf("output in sequential form: ");
    for(i=0;i<5;i++)
    {
        cout << a[i] << " ";
    }
    cout << endl;

    printf("output in ordered fashion: ");
    #pragma omp parallel for ordered 
    for(i=0;i<5;i++)
    {
        #pragma omp ordered
        {
        cout << a[i] << " ";
        }
    }
    cout << endl;


    printf("output in random order: ");
    #pragma omp parallel for
    for(i=0;i<5;i++)
    {
        cout << a[i] << " ";
    }
    cout << endl;

    return 0;
}