/*
Proc 0 asks for size of square matrix -> bcast to others.
For each block of n/p rows, each proc creates a block of n/p X n/p to send to proc 0.
This is done p times to populate the entire nXn matrix.
p here is total_nodes - 1 i.e.
mpiexec -n p' ./block_matrix_send.exe
p' = p+1. Proc 0 will only recieve from the other procs.
*/

#include<iostream>
#include<string.h>
#include<mpi.h>

using namespace std;


int main(int argc, char **argv)
{
    int my_node,total_nodes;
    MPI_Status status;
    MPI_Init(&argc,&argv);
    MPI_Comm_rank(MPI_COMM_WORLD,&my_node);
    MPI_Comm_size(MPI_COMM_WORLD,&total_nodes);

    int n;
    if(my_node == 0)
    {
        printf("Enter order of square matrix\n");
        fflush(stdout);
        scanf("%d",&n);
    }
    MPI_Bcast(&n,1,MPI_INT,0,MPI_COMM_WORLD);

    int p = total_nodes;
    int a_loc_glob[n/(p-1)][n];
    int a_glob[n][n];

    MPI_Datatype mpi_block;
    MPI_Type_vector(n/(p-1),n/(p-1),n,MPI_INT,&mpi_block);
    MPI_Type_commit(&mpi_block);
    
    for(int i=0;i<p-1;i++) //p calls 
    {
        if(my_node != 0)
        {
            int a_loc[n/(p-1)][n/(p-1)];
            for(int i1 = 0;i1<n/(p-1);i1++)
            {
                for(int i2=0;i2<n/(p-1);i2++)
                {
                    a_loc[i1][i2] = my_node;
                }
            }
            int tag = stoi(std::to_string(my_node) +"0");
            // MPI_Send(a_loc,(n/p)*(n/p),MPI_INT,0,tag,MPI_COMM_WORLD); //wrong - have to send block - create using MPI_Type_vector
            MPI_Send(&(a_loc[0][0]),n/(p-1)*n/(p-1),MPI_INT,0,tag,MPI_COMM_WORLD);
        }
    
        else if(my_node == 0)
        {
            for(int j=1;j<total_nodes;j++)
            {
                int tag = stoi(std::to_string(j)+"0");
                
                MPI_Recv(&(a_glob[i*n/(p-1)][(j-1)*n/(p-1)]),1,mpi_block,j,tag,MPI_COMM_WORLD,&status);
   
            }
        }
    }

    if(my_node == 0)
    {
        for(int v = 0;v<n;v++)
        {
            for(int w =0;w<n;w++)
            {
                cout << a_glob[v][w] << " ";
            }
            cout << endl;
        }
    }

    MPI_Finalize();

}