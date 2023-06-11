/*
Commnicating row of a matrix using MPI_Type_contiguous.
*/

#include<iostream>
#include<mpi.h>

using namespace std;


int main(int argc, char **argv)
{
    int a[3][3] = {{1,2,3},
                   {6,8,9},
                   {13,14,15}};


    int my_node,total_nodes;
    MPI_Status status;
    MPI_Init(&argc,&argv);
    
    MPI_Comm_rank(MPI_COMM_WORLD,&my_node);
    MPI_Comm_size(MPI_COMM_WORLD,&total_nodes);
    
    //Contiguous
    MPI_Datatype row_sender;
    MPI_Type_contiguous(3,MPI_INT,&row_sender);
    MPI_Type_commit(&row_sender);

    if(my_node == 0)
    {
        a[0][0] = 17;
        a[0][1]= 18;
        a[0][2] = 19;

        MPI_Send(&(a[0][0]),1,row_sender,1,34,MPI_COMM_WORLD); //if this and the next line are commented, changes made above will not be recevied by p1
    }
    if(my_node == 1) MPI_Recv(&(a[0][0]),1,row_sender,0,34,MPI_COMM_WORLD,&status);

    if(my_node ==1)
    {
        printf("For process %d the values in the matrix are:\n",my_node);
        for(int i=0;i<3;i++)
        {
            for(int j=0;j<3;j++)
            {
                cout << a[i][j] << " ";
            }
            cout << endl;
        }
    }

    

    MPI_Finalize();


}