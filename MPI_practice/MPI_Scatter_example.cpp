/*
Example of usage of MPI_Scatterv. Only the root node needs to have knowledge of the chunk sizes and displacements while receiving nodes 
can receive data in a locally declared array.
*/

#include<iostream>
#include<mpi.h>
#include<omp.h>
#include<cmath>

using namespace std;

int main(int argc , char **argv)
{
    int my_node, total_nodes;
    int n_iter = 10;
    int n_loc_iter;
    int *displacements;
    int *vecsize;
    int *x_glob;
    int *x_loc;

    MPI_Init(&argc, &argv); 
    MPI_Comm_rank(MPI_COMM_WORLD,&my_node);
    MPI_Comm_size(MPI_COMM_WORLD,&total_nodes);
    if(my_node == 0)
    {
        x_glob = new int[6]{1,2,3,4,5,6};
        displacements = new int[3];
        displacements[0] = 0;
        displacements[1] = 2;
        displacements[2] = 4;

        vecsize = new int[3];
        vecsize[0] = 2;
        vecsize[1] = 2;
        vecsize[2] = 2;
        

        x_loc = new int[2];

        MPI_Scatterv(&(x_glob[0]),vecsize,displacements,MPI_INT,&(x_loc[0]),2,MPI_INT,0,MPI_COMM_WORLD);   
        cout << x_loc[0] << " " << x_loc[1] << endl;

    }
    else
    {
        //nodes 1 and 2 don't need to have knowledge of the vecsize and displacement arrays. They can just refer to some global pointer
        //declared globally. They receive data from node 0 that has the knowledge of the data that is being sent to each proc.
        x_loc = new int[2];
        MPI_Scatterv(&(x_glob[0]),vecsize,displacements,MPI_INT,&(x_loc[0]),2,MPI_INT,0,MPI_COMM_WORLD); 
        
        // if(my_node == 2) cout << x_loc[0] << " " << x_loc[1] << endl;


    }

    

    MPI_Finalize();
    return 0;
}