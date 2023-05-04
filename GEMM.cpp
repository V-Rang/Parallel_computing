#include<iostream>
#include<vector>
#include<random>
#include<chrono>
#include<omp.h>

#define tic chrono::high_resolution_clock::now()
#define toc chrono::high_resolution_clock::now()
#define milliseconds(x) std::chrono::duration_cast<std::chrono::milliseconds>(x)
#define seconds(x) std::chrono::duration_cast<std::chrono::seconds>(x)

using Row = std::vector<double>;
using Matrix = std::vector<Row>;

using namespace std;

int main()
{

    double a[4][4] = {{1,2,3,4},
    {5,6,7,8},
    {9,10,11,12},
    {13,14,15,16}};


    // for(int i=0;i<4;i++)
    // {
    //     for(int j=0;j<4;j++)
    //     {
    //         cout << a[i][j] << " ";
    //     }
    //     cout << endl;
    // }
    // cout << endl;


    double b[4] = {1,2,3,4};

    double c[4];

    //serial mat-vec multiplication
    for(int i=0;i<4;i++)
    {
        c[i] = 0;
        for(int j = 0;j<4;j++)
        {
            c[i] += a[i][j]*b[j];
        }
    }

    // for(int i=0;i<4;i++)
    // {
    //     cout << c[i] << " ";
    // }
    // cout << endl;

    const int rows = 1000;
    const int cols = 750;
    const double minVal = 1.0;
    const double maxVal = 10.0;
    const int cols2 = 500;


    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double>dist(minVal,maxVal);

    Matrix mat(rows,Row(cols));
    Matrix bmat(cols,Row(cols2));

    
    for(int i=0;i<rows;i++)
    {
        for(int j=0;j<cols;j++)
        {
            mat[i][j] = dist(gen);
        }
    }

    for(int i=0;i<cols;i++)
    {
        for(int j=0;j<cols2;j++)
        {
            bmat[i][j] = dist(gen);
        }
    }

    //serial multiplication of rowsXcols matrix with vector

    // for(int i=0;i<rows;i++)
    // {
    //     for(int j=0;j<cols;j++)
    //     {
    //         cout << mat[i][j] << " ";
    //     }
    //     cout << endl;
    // }
    // cout << endl;

    // for(int i=0;i<cols;i++)
    // {
    //     for(int j=0;j<cols2;j++)
    //     {
    //         cout << bmat[i][j] << " ";
    //     }
    //     cout << endl;
    // }
    // cout << endl;

    Matrix cmat(rows,Row(cols2));

    auto start = tic;
    for(int i =0;i<rows;i++)
    {
        for(int j = 0;j<cols2;j++)
        {
            cmat[i][j] = 0;
            for(int k=0;k<cols;k++)
            {
                cmat[i][j] += mat[i][k]*bmat[k][j];
            }
        }
    }
    auto end = toc;
    cout<<"Time taken by serial code: "<<std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count()<< " milliseconds."<<endl;
    
    Matrix cmat_par(rows,Row(cols2));

    //parallelization of GEMM
    start = tic;
    #pragma omp parallel for
    for (int i=0;i<rows;i++)
    {

        for(int j=0;j<cols2;j++)
        {
            cmat_par[i][j] = 0;
            // #pragma omp parallel for reduction(+:cmat_par[i][j])
            #pragma omp parallel for 
            for(int k=0;k<cols;k++)
            {
                cmat_par[i][j] += mat[i][k]*bmat[k][j];
            }
        }
    }
    end = toc;
    cout<<"Time taken by parallel code: "<<std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count()<< " milliseconds."<<endl;


    // for(int i=0;i<rows;i++)
    // {
    //     for(int j=0;j<cols2;j++)
    //     {
    //         cout << cmat[i][j] << " ";
    //     }
    //     cout << endl;
    // }
    // cout << endl;

    // for(int i=0;i<rows;i++)
    // {
    //     for(int j=0;j<cols2;j++)
    //     {
    //         cout << cmat_par[i][j] << " ";
    //     }
    //     cout << endl;
    // }
    // cout << endl;




    // cout << start << " " << end << endl;
    // cout << milliseconds(end-start)<< " milliseconds"<<endl;




    // for(int i=0;i<rows;i++)
    // {
    //     for(int j=0;j<cols2;j++)
    //     {
    //         cout << cmat[i][j] << " ";
    //     }
    //     cout << endl;
    // }
    // cout << endl;



        // cmat[i] = vector<double>(cols2,0);
        // for(int j = 0;j<cols2;j++)
        // {
            // cmat[i][j] += mat[i][j]*bmat[i][j];
        // }





}