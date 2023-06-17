/*
Use of MPI_Cart:

MPI_Cart_create -> MPI_Comm_rank -> MPI_Cart_coords.
*/

#include<iostream>
#include<mpi.h>

using namespace std;

int main(int argc, char **argv)
{
    MPI_Init(&argc,&argv);

    MPI_Comm new_comm;
    int q = 2;
    int dim_sizes[3];
    dim_sizes[0] = dim_sizes[1] = dim_sizes[2] = 2;
    
    int wrap_around[3];
    wrap_around[0] = wrap_around[1] = wrap_around[2] = 1;

    int reorder = 1;

    MPI_Cart_create(MPI_COMM_WORLD,3,dim_sizes,wrap_around,reorder,&new_comm);

    int new_rank;

    MPI_Comm_rank(new_comm,&new_rank);

    int coordinates[3];

    MPI_Cart_coords(new_comm,new_rank,3,coordinates);

    printf("Process %d has coordinates = (%d,%d,%d)\n",new_rank,coordinates[0],coordinates[1],coordinates[2]);

    MPI_Finalize();

}