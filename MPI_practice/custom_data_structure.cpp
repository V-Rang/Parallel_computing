/*
custom data structure to communicate multiple variable values in MPI using MPI_Type_create_struct
*/

#include<iostream>
#include<mpi.h>

using namespace std;


void Build_derived_datatype(float *a_ptr, float *b_ptr, int *n_ptr, MPI_Datatype *mesg_mpi)
{
    int block_lengths[3];
    block_lengths[0] = block_lengths[1] = block_lengths[2] = 1;

    MPI_Datatype typelists[3];
    typelists[0] = MPI_FLOAT;
    typelists[1] = MPI_FLOAT;
    typelists[2] = MPI_INT;

    MPI_Aint displacements[3];
    displacements[0] = 0;
    MPI_Aint address;

    MPI_Get_address(a_ptr,&address);
    MPI_Aint start_address = address;


    MPI_Get_address(b_ptr,&address);
    displacements[1] = address - start_address;


    MPI_Get_address(n_ptr,&address);
    displacements[2] = address - start_address;


    MPI_Type_create_struct(3,block_lengths,displacements,typelists,mesg_mpi);

    MPI_Type_commit(mesg_mpi);

}


int main(int argc, char **argv)
{
    int my_node,total_nodes;
    float *a_ptr,*b_ptr;
    int *n_ptr;
    
    a_ptr = new float[1];
    b_ptr = new float[1];
    n_ptr = new int[1];
    MPI_Status status;
    MPI_Init(&argc,&argv);
    MPI_Comm_rank(MPI_COMM_WORLD,&my_node);
    MPI_Comm_size(MPI_COMM_WORLD,&total_nodes);

    if(my_node == 0)
    {
        printf("Enter the values of a, b and n\n");
        fflush(stdout);
        scanf("%f %f %d",a_ptr,b_ptr,n_ptr);
    }

    MPI_Datatype mpi_mesg;

    Build_derived_datatype(a_ptr,b_ptr,n_ptr,&mpi_mesg);

    MPI_Bcast(a_ptr,1,mpi_mesg,0,MPI_COMM_WORLD);

    printf("Process %d has the values of a, b and n = %f, %f, %d\n",my_node,*a_ptr,*b_ptr,*n_ptr);

    MPI_Finalize();
}