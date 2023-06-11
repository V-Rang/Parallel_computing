/*

Example of using MPI_Type_vector where blocklength is not = 1. We change all the entries in the 3 X 3 matrix
using process 0 and then communicate all the changes to process 1. 3 elements (rows) are communicated, where the
number of entries in each element = 3 (number of columns
)
MPI_Type_vector:
count = number of elements in the type
blocklength = number of entries in each element
stride = number of elements of type element_type between successive elements of new_mpi_t.
element_type: type of elements composing the derived type
new_mpi_t: MPI type of the new derived type

*/

#include<iostream>
#include<mpi.h>

using namespace std;

int main(int argc, char **argv)
{
    int a[3][3] = {{1,2,3},{6,8,9},{13,14,15}};


    int my_node,total_nodes;
    MPI_Status status;
    MPI_Init(&argc,&argv);
    
    MPI_Comm_rank(MPI_COMM_WORLD,&my_node);
    MPI_Comm_size(MPI_COMM_WORLD,&total_nodes);
    
    //Vector
    MPI_Datatype mat_sender;
    MPI_Type_vector(3,3,3,MPI_INT,&mat_sender);
    MPI_Type_commit(&mat_sender);

    if(my_node == 0)
    {
        
        a[0][0] = 10;
        a[0][1] = 20;
        a[0][2] = 30;

        a[1][0] = 60;
        a[1][1] = 80;
        a[1][2] = 90;

        a[2][0] = 130;
        a[2][1] = 140;
        a[2][2] = 150;

        MPI_Send(&(a[0]),3,mat_sender,1,34,MPI_COMM_WORLD); //if this and the next line are commented, changes made above will not be recevied by p1
    }

    if(my_node == 1) MPI_Recv(&(a[0]),3,mat_sender,0,34,MPI_COMM_WORLD,&status);

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