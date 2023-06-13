/*
Using MPI_Type_vector to send column of a matrix which is then stored in the row of the matrix for process 1.
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
    
    int a[3][3] = {{1,2,3},
                   {4,5,6},
                   {7,8,9}};
   
   
    if(my_node == 0)
    {
        a[0][1] = 20;
        a[1][1] = 50;
        a[2][1] = 80;
        MPI_Datatype mpi_vect;
        MPI_Type_vector(3,1,3,MPI_INT,&mpi_vect);
        MPI_Type_commit(&mpi_vect);
        MPI_Send(&(a[0][1]),1,mpi_vect,1,34,MPI_COMM_WORLD);

    }
    

    if(my_node == 1)
    {
        MPI_Recv(&(a[1][0]),3,MPI_INT,0,34,MPI_COMM_WORLD,&status);
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