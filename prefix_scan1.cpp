/*
Prefix scanning using two algorithms:
1. Inclusive scan using Hillis-Steele alogirthm.
2. Exclusive scan performed using Blelloch algorithm.

Run using:
g++ -fopenmp -o exec prefix_scan1.cpp
./exec

*/
#include<iostream>
#include<omp.h>
#include<chrono>
#include<random>
#include<cmath>

#define tic chrono::high_resolution_clock::now()
#define toc chrono::high_resolution_clock::now()

#define milliseconds(x) std::chrono::duration_cast<std::chrono::milliseconds>(x)
#define microseconds(x) std::chrono::duration_cast<std::chrono::microseconds>(x)     


using namespace std;


int main()
{

    const double minval  = 0.0;
    const double maxval = 1.0; 
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double>dist(minval,maxval);

    int n = (int)8;
    double a[n],a_ser_incl[n],a2[n],a_ser_excl[n],a3[n];
    int i,j;
    //creating an array a: 1 2 3 4 5 6 7 8
    //a_ser_inc is used to calculate the inclusive scan output array seraially.
    //a3 is same as a and is used for the exclusive scan case.
    for(i=0;i<n;i++)
    {
        // a[i] = dist(gen);
        a[i] = i+1;
        a_ser_incl[i] = a[i];
        a3[i] = a[i];
    }
    
    printf("The array to be scanned is: ");
    for(i=0;i<n;i++)
    {
        cout << a[i] << " ";
    }
    cout << endl;


    
    for(i=1;i<n;i++)
    {
        a_ser_incl[i] += a_ser_incl[i-1];
    }    

    //inclusive scan using Hillis-Steele algorithm - D=O(logn) W=O(nlogn)
    for(i=0;i<floor(log2(n));i++)
    {
        double a2[n] = {0};
        for(j=0;j<n;j++) a2[j] = a[j];

        #pragma omp parallel for private(j) shared(i,n,a,a2) default(none)
        for(j=0;j<(int)(n-pow(2,i));j++)
        {
            a[(int)(j+pow(2,i))] += a2[j];
        }
    }


    printf("Inclsive scan performed serially: ");
    for(i = 0;i<n;i++)
    {
        cout << a[i] << " ";
    }
    cout << endl;

    printf("Inclsive scan performed parallely using Hillis-Steele Algorithm: ");
    for(i = 0;i<n;i++)
    {
        cout << a_ser_incl[i] << " ";
    }
    cout << endl;


    //exclusive scan using Blelloch algorithm - D=O(logn) W=O(n)
    a_ser_excl[0] = 0;
    for(i=1;i<n;i++)
    {
        a_ser_excl[i] = a_ser_excl[i-1] + a3[i-1];
    }

    for(i=1;i<=floor(log2(n));i++)
    {
        for(j=pow(2,i)-1;j<n;j+= pow(2,i))
        {
            a3[j] += a3[(int)(j-pow(2,i-1))];
        }
    }

    a3[n-1] = 0;
    double temp;

    for(i=floor(log2(n));i>=1;i--)
    {
        for(j=pow(2,i)-1;j<n;j += pow(2,i))
        {
            temp = a3[(int)(j-pow(2,i-1))];
            a3[(int)(j-pow(2,i-1))] = a3[j];
            a3[j] += temp;
        }
    }

    printf("Exclusive scan performed serially: ");
    for(i = 0;i<n;i++)
    {
        cout << a_ser_excl[i] << " ";
    }
    cout << endl;

    printf("Exclusive scan performed parallely using Blelloch Algorithm: ");
    for(i = 0;i<n;i++)
    {
        cout << a3[i] << " ";
    }
    cout << endl;





    // a_ser[0] = 0;
    // for(i=1;i<n;i++)
    // {
    //     a_ser[i] = a_ser[i-1] + a[i-1];
    // }   


    // for(i=1;i<=floor(log2(n));i++)
    // {
    //     #pragma omp parallel for  private(j,i) shared(a,n) default(none) 
    //     for(j=0;j<n;j++)
    //     {
    //         if(j>=pow(2,i))
    //         {
    //             a[j] = a[(int)(j-pow(2,i-1))] + a[j];
    //         }
    //     }
    // }


    // int non_zero_dif = 0;
    // for(i = 0;i<n;i++)
    // {
    //     if( (int)(a[i] - a_ser[i]) != 0) non_zero_dif += 1;
    // }
    // printf("Number of indices where serial and parallel sum have different values = %d\n",non_zero_dif);
    
    // for(i = 0;i<n;i++)
    // {
    //     cout << a[i] << " ";
    // }
    // cout << endl;


    // for(i = 0;i<n;i++)
    // {
    //     cout << a_ser[i] << " ";
    // }
    // cout << endl;


    return 0;
}