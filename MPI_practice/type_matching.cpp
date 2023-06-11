/*
Derived data types are just a sequence of pairs:
{(t0,d0) ... (tn-1,dn-1)}
t's -> Basic MPI type
d's -> Displacements

For MPI_Send and MPI_Recv:
1. type signature (t0,..tn-1) must match.
2. send_count <= recv_count

If above two are true, it is possible to communicate data.
In this example, col 1 of a is modified by process 0, process 1 modifies row 1 of a using the data. 
*/

#include<iostream>
#include<mpi.h>

using namespace std;

int main(int argc, char **argv)
{
    int n = 3;
    int a[3][3] = {{1,2,3},
                   {6,8,9},
                   {13,14,15}};

    
    int my_node,total_nodes;
    MPI_Status status;
    MPI_Init(&argc,&argv);
    
    MPI_Comm_rank(MPI_COMM_WORLD,&my_node);
    MPI_Comm_size(MPI_COMM_WORLD,&total_nodes);
    
    MPI_Datatype col_sender;
   
    MPI_Type_vector(3,1,3,MPI_INT,&col_sender);
    MPI_Type_commit(&col_sender);

    if(my_node == 0)
    {
        a[0][1] = 10;
        a[1][1] = 20;
        a[2][1] = 30;
        MPI_Send(&(a[0][1]),1,col_sender,1,34,MPI_COMM_WORLD);

    }
    else if(my_node == 1)
    {
        MPI_Recv(&(a[1][0]),3,MPI_INT,0,34,MPI_COMM_WORLD,&status);
    }


    if(my_node == 1)
    {
        printf("Matrix for process %d\n",my_node);
        for(int i=0;i<n;i++)
        {
            for(int j=0;j<n;j++)
            {
                cout << a[i][j] << " ";
            }
            cout << endl;
        }
    }




       

    MPI_Finalize();


}