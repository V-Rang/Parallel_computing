#include<iostream>
#include<omp.h>
#include<random>


using namespace std;

int main()
{
    omp_set_num_threads(5);
    // #pragma omp parallel
    // {
    //     printf("Hello from thread %d\n",omp_get_thread_num());
    // }

    const double minval = 1.0;
    const double maxval = 100.0;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double>dist(minval,maxval);

    int i,j,k;

    int p = (int)1e3;
    int q = (int)10;
    int r = (int)1e2;

    double a_mat[p][q] = {0};
    double b_mat[q][r] = {0};
    double c_mat[p][r] = {0};
    
    //Initializing matrices
    for(i=0;i<p;i++)
    {
        for(j=0;j<q;j++)
        {
            // a_mat[i][j] = i+2*j;
            a_mat[i][j] = dist(gen);
        }
    }

    for(i=0;i<q;i++)
    {
        for(j=0;j<r;j++)
        {
            // b_mat[i][j] = 3*i+j;
            b_mat[i][j] = dist(gen);
        }
    }


    // //serial mat-mat mult
    // for(k=0;k<r;k++) // rtimes for each column in c_mat
    // {
    //     double c_loc[p] = {0}; // for kth col of C, 
    //     for(i=0;i<p;i++)
    //     {
    //         for(j=0;j<q;j++)
    //         {
    //             c_loc[i] += a_mat[i][j]*b_mat[j][k];
    //         }
    //     }

    //     for(i=0;i<p;i++)
    //     {
    //         c_mat[i][k] = c_loc[i];
    //     }
    // }

    // serial mat-mat mult
    for(k=0;k<r;k++) // rtimes for each column in c_mat
    { 
        for(i=0;i<p;i++) //within the kth col of c_mat, the ith row is the sum of q values
        { 
            for(j=0;j<q;j++) // summing product over q cols to get the c_mat[ith row][kth col] value.
            {
                c_mat[i][k] += a_mat[i][j]*b_mat[j][k];
            }
        }
    }

    //parallel mat-mat mult
    omp_set_num_threads(5);

    double c_glob[p][r] = {0};
    #pragma omp parallel private(k,i,j) shared(a_mat,b_mat,c_glob,r,p,q) default(none)
    {
        #pragma omp for 
        for(k=0;k<r;k++)
        {
            for(i=0;i<p;i++)
            {   
                for(j=0;j<q;j++)
                {
                    c_glob[i][k] += a_mat[i][j]*b_mat[j][k];
                }
            }
        }
    }

    // printf("Matrix A:\n");
    // for(i=0;i<p;i++)
    // {
    //     for(j=0;j<q;j++)
    //     {
    //         cout << a_mat[i][j] << " ";
    //     }
    //     cout << endl;
    // }

    // printf("Matrix B:\n");
    // for(i=0;i<q;i++)
    // {
    //     for(j=0;j<r;j++)
    //     {
    //         cout << b_mat[i][j] << " ";
    //     }
    //     cout << endl;
    // }


    // printf("Serial Matrix C:\n");
    // for(i=0;i<p;i++)
    // {
    //     for(j=0;j<r;j++)
    //     {
    //         cout << c_mat[i][j] << " ";
    //     }
    //     cout << endl;
    // }



    // printf("Parallel Matrix C:\n");
    // for(i=0;i<p;i++)
    // {
    //     for(j=0;j<r;j++)
    //     {
    //         cout << c_glob[i][j] << " ";
    //     }
    //     cout << endl;
    // }

    //discrepancy between serial and matrix calculation
    int counter = 0;
    for(i=0;i<p;i++)
    {
        for(j=0;j<r;j++)
        {
            if(abs(c_mat[i][j] - c_glob[i][j]) > 1e-4 ) counter += 1;
        }
    }
    printf("Number of values different in serial and parallel mat-mat mult = %d\n",counter);
    return 0;
}