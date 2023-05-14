/*
invalid exit from OpenMP structured block.
Reason: Invalid branch out of the block. 
*/
#include<iostream>
#include<omp.h>
#include<cmath>
#define MAX 100

using namespace std;


void test_func(int a[],int n)
{ 
    int i;
    omp_set_num_threads(5);  
    #pragma omp for
    for(i=0;i<n;i++)
    {
        if(a[i] > 3)
        {
            return;
        }
        a[i] = pow(a[i],2);
    }
    return;
    
}

int main()
{
    int a[5] = {0};
    
    for(i=0;i<5;i++)
    {
        a[i] = i+1;
    }

    int n = sizeof(a)/sizeof(int);
    test_func(a,n);

    for(i=0;i<5;i++)
    {
        cout << a[i] << " ";
    }
    cout << endl;
    return 0;
}