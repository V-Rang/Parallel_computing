/*
Sending contiguous block of memory from one proc to another.
C++ is row major. So rows are contiguous blocks of memory and so can be sent.
*/

#include<iostream>
#include<mpi.h>


using namespace std;

int main(int argc, char **argv)
{
    int a[2][2] = {{4,5},
                   {6,7}};

    int b[2];

    int my_node,total_nodes;
    MPI_Status status;
    MPI_Init(&argc,&argv);
    MPI_Comm_rank(MPI_COMM_WORLD,&my_node);
    MPI_Comm_size(MPI_COMM_WORLD,&total_nodes);
    if(my_node == 0)
    {
        MPI_Send(&(a[0][0]),2,MPI_INT,1,34,MPI_COMM_WORLD);
    }
    else
    {
        // int b[2];
        MPI_Recv(&(b[0]),2,MPI_INT,0,34,MPI_COMM_WORLD,&status);
    }

    if(my_node == 1)
    {
        for(int i=0;i<2;i++)
        {
            cout << b[i] << " ";
        }
        cout << endl;
    }
    MPI_Finalize();

}