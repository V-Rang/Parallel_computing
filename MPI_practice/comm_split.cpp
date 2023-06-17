/*
MPI_Comm_split
*/
#include<iostream>
#include<mpi.h>

using namespace std;

int main(int argc, char **argv)
{
    int my_node,total_nodes;
    int r = 10;
    MPI_Init(&argc,&argv);
    MPI_Comm_rank(MPI_COMM_WORLD,&my_node);
    MPI_Comm_size(MPI_COMM_WORLD,&total_nodes);
    
    MPI_Comm example_comm;
    int q = 3;
    int my_row = my_node/q;

    MPI_Comm_split(MPI_COMM_WORLD,my_row,my_node,&example_comm);

    if(my_node == 8) r = 4;

    MPI_Bcast(&r,1,MPI_INT,2,example_comm); 

    /*
    all instances of example_comm have 3 procs each - with id - 0, 1 and 2. 
    If bcast with id > 2 -> get error.
    if bcast with 0, 1 or 2. change will be reflected in 3 procs only.
    */
    printf("Process %d has the value = %d\n",my_node,r);





    MPI_Finalize();
}