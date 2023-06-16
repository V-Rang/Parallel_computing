/*
Two important inferences regarding MPI_Reduce:
1. The root proc must also have the array being reduced as MPI_Reduce reduces values from ALL procs.
2. The addresses of the send and recv buffer have to be in the format &( []), NOT &__ , the latter gives error.
*/

#include<iostream>
#include<mpi.h>

using namespace std;

int main(int argc, char **argv)
{
    int my_node,total_nodes;
    MPI_Init(&argc,&argv);
    MPI_Comm_rank(MPI_COMM_WORLD,&my_node);
    MPI_Comm_size(MPI_COMM_WORLD,&total_nodes);
    int n = 4;
    int p = total_nodes;

    int *y_glob;
    int *y_loc;

    if(my_node != 0)
    {
        y_loc = new int[n/(p-1)];
        for(int i=0;i<n/(p-1);i++)
        {
            y_loc[i] = my_node + 2*i;
        }

        printf("For process %d\n",my_node);
        for(int i=0;i<n/(p-1);i++)
        {
            cout << y_loc[i] << " ";
        }
        cout << endl;
    }
    else
    {
        y_loc = new int[n/(p-1)];
        for(int i=0;i<n/(p-1);i++) //VVI, Proc 0 should have a y_loc as well as MPI_Reduce reduces values from ALL procs
        {
            y_loc[i] = 0;
        }
        
        y_glob = new int[n/(p-1)];

        
    }


    MPI_Reduce(&(y_loc[0]),&(y_glob[0]),n/(p-1),MPI_INT,MPI_SUM,0,MPI_COMM_WORLD);
    
    // MPI_Reduce(&y_loc,&y_glob,n/(p-1),MPI_INT,MPI_SUM,0,MPI_COMM_WORLD); //wrong, have to use code in above line.
    

    if(my_node == 0)    
    {
        printf("Process %d has the reduced values\n",my_node);
        for(int i=0;i<(n/(p-1));i++)
        {
            cout << y_glob[i] << " ";
        }
        cout << endl;

    }


    MPI_Finalize();

}