/*
Example of changing values of x,y and z on 3 procs depending on the order of execution of different statements. 
Execute using mpiexec -n 3 ./random_values.exe
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
    MPI_Bcast()
    int x,y,z;
    switch(my_node)
    {
        case 0: x = 0;y=1;z=2;
        MPI_Bcast(&x,1,MPI_INT,0,MPI_COMM_WORLD);
        MPI_Send(&y,1,MPI_INT,2,43,MPI_COMM_WORLD);
        MPI_Bcast(&z,1,MPI_INT,1,MPI_COMM_WORLD);
        break;

        case 1: x=3;y=4;z=5;
        MPI_Bcast(&x,1,MPI_INT,0,MPI_COMM_WORLD);
        MPI_Bcast(&y,1,MPI_INT,1,MPI_COMM_WORLD);
        break;

        case 2: x=6;y=7;z=8;
        MPI_Bcast(&z,1,MPI_INT,0,MPI_COMM_WORLD);
        MPI_Recv(&x,1,MPI_INT,0,43,MPI_COMM_WORLD,&status);
        MPI_Bcast(&y,1,MPI_INT,1,MPI_COMM_WORLD);
        break;
    }

    printf("Process: %d has the values: x = %d, y = %d, z = %d\n",my_node,x,y,z);

    MPI_Finalize();


}