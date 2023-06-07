/*
Send and Recv integer arrays from process 0 to process 1.
use mpiexec -n 2 ./executable.exe 
*/

#include<iostream>
#include<mpi.h>

using namespace std;

int main(int argc, char **argv)
{
    int my_node,total_nodes;
    MPI_Status status;
    MPI_Init(&argc,&argv);
    MPI_Comm_rank(MPI_COMM_WORLD,&my_node);
    MPI_Comm_size(MPI_COMM_WORLD,&total_nodes);

    if(my_node == 0)
    {
        int *x = new int[4];
        for(int i=0;i<4;i++) x[i] = i+1;

        MPI_Send(x,4,MPI_INT,1,1,MPI_COMM_WORLD);
    }
    else
    {
        int *y = new int[4];
        MPI_Recv(y,4,MPI_INT,0,1,MPI_COMM_WORLD,&status);
        for(int i=0;i<4;i++) cout << y[i] << " ";
        cout << endl;
    }

    MPI_Finalize();
}
