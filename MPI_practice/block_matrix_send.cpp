/*
Process 0 reads in number of rows, bcasts it to all others.
Process 0 reads in n/p rows at a time, sends it in n/p X n/p blocks to all the other procs.

The procs can either read it in to a n/p X n/p block or into a n X n/p block which is populated successively for each call to read in the n/p block of rows by proces 0.

Send and receive done in single call (1 count) to MPI_Type_vector.

Previous error: MPI_Type_vector(count) atttribute has to be n/p, NOT 1. since the number of blocks to be communicated is n/p.

Blocklength for send and recive both will be n/p.
Stride for send will be n and for recv will be n/p.

*/

#include<iostream>
#include<mpi.h>
#include<string.h>


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
        printf("Enter number of rows of square matrix\n");
        fflush(stdout);
        scanf("%d",&n);
    }
    MPI_Bcast(&n,1,MPI_INT,0,MPI_COMM_WORLD);
    int p = total_nodes;
    MPI_Datatype mpi_block;
    // MPI_Type_vector()
    MPI_Type_vector(n/(p-1),n/(p-1),n,MPI_INT,&mpi_block);
    MPI_Type_commit(&mpi_block);

    MPI_Datatype mpi_block2;
    MPI_Type_vector(n/(p-1),n/(p-1),n/(p-1),MPI_INT,&mpi_block2);
    MPI_Type_commit(&mpi_block2);

    int a_loc[n][n/(p-1)]; //for every proc other than 0, a block of n x n/p. For each block of n/p rows, succesive rows are populated
    int a_loc2[n/(p-1)][n/(p-1)];//for every proc other than 0, a block of n/p X n/p. For each block of n/p rows, these are populated.

    for(int i=0;i<p-1;i++) // for each block of n/p rows
    {
        if(my_node == 0)
        {
            int a_loc_glob[n/(p-1)][n]; //n/p X n block

            for(int j=0;j<n/(p-1);j++)
            {
                printf("Enter %d elements for row %d\n",n,j + i*(n/(p-1)));
                fflush(stdout);
                for(int k=0;k<n;k++)
                {
                    int temp;
                    // scanf("%d",a_loc_glob[j][k]);
                    scanf("%d",&temp);
                    a_loc_glob[j][k] = temp;
                }
            }// finished making n/p X n. Now communicate this as n/p X n/p blocks to all the other processes
            
            for(int j=1;j<p;j++)
            {
                int tag = stoi(std::to_string(j) + "0"+std::to_string(i));
                MPI_Send(&(a_loc_glob[0][(j-1)*(n/(p-1))]),1,mpi_block,j,tag,MPI_COMM_WORLD);
            }    
        }
        else
        {
            int tag = stoi(std::to_string(my_node) + "0"+std::to_string(i));
            MPI_Recv(&(a_loc[i*(n/(p-1))][0]),1,mpi_block2,0,tag,MPI_COMM_WORLD,&status); //successively populate rows of the nX(n/p) block designated to every proc.
           
            // MPI_Recv(a_loc2,1,mpi_block2,0,tag,MPI_COMM_WORLD,&status); //can uncomment and the following block of code to see n/p X n/p communicated to every proc
            
            // if(my_node == 2)
            // {
            //     for(int v = 0;v<n/(p-1);v++)
            //     {
            //         for(int w = 0;w<n/(p-1);w++)
            //         {
            //             cout << a_loc2[v][w] << " ";
            //         }
            //         cout << endl;
            //     }
            // }

        }
    }
    if(my_node == 4) //can change to see the block of n X n/p recieved by every proc from proc 0.
    {
        printf("For process %d\n",my_node);
        for(int i=0;i<n;i++)
        {
            for(int j=0;j<n/(p-1);j++)
            {
                cout << a_loc[i][j] << " ";
            }
            cout << endl;
        }
    }
    
    MPI_Finalize();

}