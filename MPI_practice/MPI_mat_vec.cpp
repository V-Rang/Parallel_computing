/*
Scatter and Gather for mat-vec multiplication
matrix of dimension m X n.
p = number of processes
Code works when m and n are divisible by p.
*/


#include<iostream>
#include<mpi.h>

using namespace std;


int main(int argc, char **argv)
{
    int m,n,p; //m X n matrix, p = no. of procs
    m = 4;
    n = 4;

    int a[m][n] = {{1,2,3,4},
                   {5,6,7,8},
                   {9,10,11,12},
                   {13,14,15,16}};

    // cout << A[1][2] << endl;

    int x[n] = {1,2,3,4};
  
    int my_node,total_nodes;
    MPI_Init(&argc,&argv);
    MPI_Comm_rank(MPI_COMM_WORLD,&my_node);
    MPI_Comm_size(MPI_COMM_WORLD,&total_nodes);

    p = total_nodes;

    int a_loc[m/p][n];
    int x_loc[n/p];
    int y_loc[m/p];

    int y[m];

    MPI_Scatter(a,m*n/p,MPI_INT,a_loc,m*n/p,MPI_INT,0,MPI_COMM_WORLD); //every proc has m/p rows of a in a_loc;
    // printf("Process %d has the a values:\n",my_node);
    // for(int j = 0;j<m/p;j++)
    // {
        
    //     for(int i=0;i<n;i++)
    //     {
    //         cout << a_loc[j][i] << " ";
    //     }
    //     cout << endl;
    // }

    //store results for each proc in y_loc. Then Gather in y at end.
    for(int i = 0;i<m/p;i++)
    {
        y_loc[i] = 0;
        for(int j=0;j<n;j++)
        {
            y_loc[i] += a_loc[i][j]*x[j];
        }
    }

    MPI_Gather(y_loc,m/p,MPI_INT,y,m/p,MPI_INT,0,MPI_COMM_WORLD); //gathering all the y_loc from all procs to store in y

    if(my_node == 0)
    {
        for(int i=0;i<m;i++)
        {
            cout << y[i] << " ";
        }
        cout << endl;
    }

    //M-2: for each set of m/p rows -> send n/p cols to the other procs to multiply with their n/p number of x_vals (NOT WORKING)
    
    // MPI_Scatter(x,n/p,MPI_INT,x_loc,n/p,MPI_INT,0,MPI_COMM_WORLD); //every proc has n/p values of x in x_loc;
    // int y_loc2[m/p];

    // for(int j=0;j<total_nodes;j++) // each node has to create its copy of y_loc2 which will then be gathered to create y.
    // {
    //     int a_loc2[m/p][n/p];
    //     MPI_Scatter(a_loc,(m/p)*(n/p),MPI_DOUBLE,a_loc2,(m/p)*(n/p),MPI_DOUBLE,j,MPI_COMM_WORLD); //I think a_loc2 will be ovewritten as every process has only 1 copy of aloc2
    //     for(int i=0;i<m/p;i++) //for each row in y_loc
    //     {
    //         y_loc2[i] = 0;
    //         for(int k = 0; k<total_nodes;k++) //sum over each process' local copy
    //         {
    //             y_loc2[i] += a_loc2[i][k]*x_loc[k]; //dot product
    //         }
    //     }
    // }

    // MPI_Gather(y_loc2,m/p,MPI_DOUBLE,y,m/p,MPI_DOUBLE,0,MPI_COMM_WORLD);

    // if(my_node == 0)
    // {
    //     for(int i=0;i<m;i++)
    //     {
    //         cout << y[i] << " ";
    //     }
    //     cout << endl;
    // }

    MPI_Finalize();

}