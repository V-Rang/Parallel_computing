#include<iostream>
#include<omp.h>
#include<random>

using namespace std;

int main()
{

    int m = (int)1e4; //rows
    int n = (int)1e2; //cols
    double a_mat[m][n] = {0};
    double b_vec[n] = {0};
    double c_vec[m] = {0};
    int i,j;

    const double minval = 1.0;
    const double maxval = 100.0;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double>dist(minval,maxval);


    for(i=0;i<m;i++)
    {
        for(j=0;j<n;j++)
        {
            // a_mat[i][j] = dist(gen);
            a_mat[i][j] = i + 2*j;
        }
    }

    for(i=0;i<n;i++)
    {
        // b_vec[i] = dist(gen);
        b_vec[i] = i*i;
    }

    //serial mat-vec
    for(i=0;i<m;i++)
    {
        for(j=0;j<n;j++)
        {
            c_vec[i] += a_mat[i][j]*b_vec[j];
        }
    }

    //parallel mat-vec
    omp_set_num_threads(3);
    double c_glob[m] = {0};
    double c_loc[m];
    int k;

    #pragma omp parallel private(j,i,c_loc,k) shared(a_mat,b_vec,n,m,c_glob) default(none)
    {   
        #pragma omp for schedule (static,2)
        for (j=0;j<n;j++) //traverse columns
        {
            c_loc[m] = {0};
            for(i=0;i<m;i++) //traverse rows in each column
            {
                c_loc[i] = a_mat[i][j]*b_vec[j];
            }    
            for(i=0;i<m;i++)
            {
                #pragma omp atomic
                c_glob[i] += c_loc[i];
            }
        }
        
    }


    //Matrix A:
    // printf("Matrix A: \n");
    // for(i=0;i<m;i++)
    // {
    //     for(j=0;j<n;j++)
    //     {
    //         cout << a_mat[i][j] << " ";
    //     }
    //     cout << endl;
    // }
    // //Vector b:
    // printf("Vector b: \n");
    // for(i=0;i<n;i++)
    // {
    //     cout << b_vec[i] << " ";
    // }
    // cout << endl;

    // //Vector c
    // printf("Vector c calculated serially: \n");
    // for(i=0;i<m;i++)
    // {
    //     cout << c_vec[i] << " ";
    // }
    // cout << endl;

    // //Vector c parallel
    // printf("Vector c calculated parallely: \n");
    // for(i=0;i<m;i++)
    // {
    //     cout << c_glob[i] << " ";
    // }
    // cout << endl;

    int counter = 0;
    for(i=0;i<m;i++)
    {
        if(  abs(c_vec[i] - c_glob[i]) > 1e-4 ) counter += 1;
    }

    printf("No. of values that are different between parallel and serial execution = %d\n",counter);

    return 0;
}