/*
Creating new communicator of 3 procs 0,1 and 2 and broadcasting info from proc 0 to procs in group underlying this
communicator only. 
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
    int q = total_nodes;
    int *process_ranks = new int[3];
    for(int i=0;i<3;i++)
    {
        process_ranks[i] = i;
    }

    MPI_Group group_world;
    MPI_Group first_row_group;
    MPI_Comm first_row_comm;
    MPI_Comm_group(MPI_COMM_WORLD,&group_world);
    MPI_Group_incl(group_world,3,process_ranks,&first_row_group);
    
    MPI_Comm_create(MPI_COMM_WORLD,first_row_group,&first_row_comm);
    

    if(my_node == 0) r = 4; //revised value of r, only reflected in procs belonging to new communicator - 0, 1 and 2.

    if(my_node < 3)
    {
        MPI_Bcast(&r,1,MPI_INT,0,first_row_comm);
    }

    printf("Process %d has the value of r = %d\n",my_node,r);



    MPI_Finalize();
}