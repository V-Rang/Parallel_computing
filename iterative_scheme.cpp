/*
Objective: perform an iterative scheme on a matrix: 
j: col index
i: row index

a[i][j] = C1 * a[i][j] + C2 * ( a[i-1][j] + a[i+1][j] + a[i][j-1] + a[i][j+1] )
C1, C2 are constants.
Do it serially, parallely (skewing) and parallely (blocking)

skewing: use fact that calculations along a diagonal can be done in parallel. diagonal: towards north-east in the scheme used.
*/

#include<iostream>
#include<omp.h>
#include<random>
#include<algorithm>
using namespace std;




// struct point
// {
//     double element = 5.0;
//     int *index;
//     int num_cols;
// } *x;


int main()
{


    const double minval = 1.0;
    const double maxval = 100.0;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double>dist(minval,maxval);

    omp_set_num_threads(2);

    // export OMP_SCHEDULE=static,3;
    // #pragma omp parallel for schedule(runtime)
    // for(int i=0;i<10;i++)
    // {
    //     printf("Thread %d is running iteration %d\n",omp_get_thread_num(),i);
    // }

    int n = 10;
    double a[n][n] = {0.0};
    double a2[n][n] = {0.0};
    double a3[n][n] = {0.0};

    int i,j;

    for(j=0;j<n;j++)
    {
        for(i=0;i<n;i++)
        {
            // a[i][j] = 0.5*a[i][j] + 0.125*(   a[i-1][j] + a[i+1][j] + a[i][j-1] + a[i][j+1] );
            a[i][j] = dist(gen); //matrix for serial modification
            a2[i][j] = a[i][j]; // matrix for parallel - skewing
            a3[i][j] = a[i][j]; // matrix for parallel - blocking
        }
    }

    // printf("Initial matrix:\n");
    // for(i=0;i<n;i++)
    // {
    //     for(j=0;j<n;j++)
    //     {
    //         cout << a[i][j] << " ";
    //     }
    //     cout << endl;
    // }
    // cout << endl;


    // modification in serial
    for(j=1;j<n-1;j++)
    {
        for(i=1;i<n-1;i++)
        {
            a[i][j] = 0.5*a[i][j] + 0.125*(   a[i-1][j] + a[i+1][j] + a[i][j-1] + a[i][j+1] );
        }
    }

    // printf("Matrix post serial modification:\n");
    // for(i=0;i<n;i++)
    // {
    //     for(j=0;j<n;j++)
    //     {
    //         cout << a[i][j] << " ";
    //     }
    //     cout << endl;
    // }

    //modification in parallel (skewing)
    for(j=4;j<=2*n-2; j++)
    {
        #pragma omp parallel for private(i) shared(a2,n,j) default(none) //firstprivate or shared j gives no error but private j does. Why?
        for(i= max(2,j-n+1); i<=min(n-1,j-2); i++)                                     // I guess private loses j value from outer loop when it enters inner loop
        {
            a2[i-1][j-i-1] = 0.5*a2[i-1][j-i-1] + 0.125*( a2[i-2][j-i-1] + a2[i][j-i-1] + a2[i-1][j-i-2] + a2[i-1][j-i] );
        }
    }

    // printf("Matrix post parallel modification:\n");
    // for(i=0;i<n;i++)
    // {
    //     for(j=0;j<n;j++)
    //     {
    //         cout << a2[i][j] << " ";
    //     }
    //     cout << endl;
    // }


    //modification in parallel (blocking)
    // int bsize = 2; // block size:1 to n-2 
    // int p;
    // int tid;
    // omp_set_num_threads(2);
    // // for(p=0;p<n/bsize;p++)
    // for(p=1;p<n/bsize;p++)
    // {
    //     // for(i=p*bsize;i<(p+1)*bsize;i++)
    //     for(i=p*bsize-1 ; i< p*bsize + 1; i++)
    //     {
    //         // #pragma omp parallel for private(j,tid) shared(i,a3,n) default(none)
    //         #pragma omp parallel 
    //         for(j=1;j<=n-1;j++)
    //         {
    //             tid = omp_get_thread_num();
    //             if(tid == 0)
    //             {
    //                 a3[i][j] = 0.5*a3[i][j] + 0.125*(   a3[i-1][j] + a3[i+1][j] + a3[i][j-1] + a3[i][j+1] );
    //             }
    //             else if(tid == 1)
    //             {
    //                 #pragma omp barrier
    //                 a3[i][j] = 0.5*a3[i][j] + 0.125*( a3[i-1][j] + a3[i+1][j] + a3[i][j-1] + a3[i][j+1] );
    //             }
    //         }
    //     }
    // }

    //modification in parallel (blocking)
    // for(j=4;j<=2*n-1;j++)
    // {

    //     for(i = max(2,j-n+1);i<=min(n-1,j-2);i++)
    //     {
    //         a[i-1][j-i-1] =  
    //     }
    // }


    //modification in parallel (blocking)
    for(j=1;j<=n-2;j++)
    {
        #pragma omp parallel for ordered
        for(i=1;i<=n-2;i++)
        {
            #pragma omp ordered
            {
                a3[i][j] = 0.5*a3[i][j] + 0.125*(   a3[i-1][j] + a3[i+1][j] + a3[i][j-1] + a3[i][j+1] );
            }
        }
    }

    printf("No of values that are different:\n");
    int counter = 0;
    // int x_index = -1;
    // int y_index = -1;
    vector<int>x_inds;
    vector<int>y_inds;
    // #pragma omp parallel for reduction(+:counter)
    for(i=0;i<n;i++)
    {
        for(j=0;j<n;j++)
        {
            if( abs(a[i][j] - a3[i][j]) > 1e-6)
            {
                counter += 1;
                x_inds.push_back(i);
                y_inds.push_back(j);
            }
        }
    }
    cout << counter << endl;
    
    if(counter != 0 )
    {
        for(i=0;i<counter;i++)
        {
            cout << "The inidices are: " << x_inds[i] << " " << y_inds[i] << endl;
            cout << "The values for the serial and modified cases are: "<< a[x_inds[i]][y_inds[i]] << " " <<  a3[x_inds[i]][y_inds[i]] << endl;
        }
    }

    // printf("Matrix post serial modification:\n");
    // for(i=0;i<n;i++)
    // {
    //     for(j=0;j<n;j++)
    //     {
    //         cout << a[i][j] << " ";
    //     }
    //     cout << endl;
    // }

    // printf("Matrix post parallel modification:\n");
    // for(i=0;i<n;i++)
    // {
    //     for(j=0;j<n;j++)
    //     {
    //         cout << a2[i][j] << " ";
    //     }
    //     cout << endl;
    // }
    return 0;
}



