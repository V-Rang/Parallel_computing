/*
MPI_Type_vector:
Parameters for MPI_Type_vector:
1. int count - no. of blocks
2. int blocklength
3. int stride

MPI_Type_indexed
Parameters for MPI_Type_indexed:
1. int count - no. of blocks
2. int blocklengths[]
3. int displacements[]

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

    int a[4][4] = {{1,2,3,4},{5,6,7,8},{9,10,11,12},{13,14,15,16}};
    int a_loc[3][2]; //2 X2  to receive one block of 2 x 2
    
    // int a_loc2[8]; // 8 length array to receive two blocks of 2 X2
    // int a_loc3[2][4]; // 2X4 array to recieve two blocks of 2X2, the 8 values are reshaped to fill in the 2X4 array.
    // int a_loc4[4][2]; // 4X2 array to recieve two blocks of 2X2, the 8 values are reshaped to fill in the 4X2 array.
  
    MPI_Datatype mpi_vect;

    int blocklength = 2;

    MPI_Type_vector(3,blocklength,4,MPI_INT,&mpi_vect);
    MPI_Type_commit(&mpi_vect);

    if(my_node == 0)
    {
        a[0][0] = 10;
        a[1][0] = 50;
        a[2][0] = 90;
        a[3][0] = 130;
        MPI_Send(&(a[0][0]),1,mpi_vect,1,34,MPI_COMM_WORLD);
    }

    if(my_node == 1)
    {
        MPI_Datatype mpi_vect2;
        // MPI_Type_vector(3,blocklength,2,mpi_vect,&mpi_vect2); //doesn't work

        MPI_Type_vector(3,blocklength,2,MPI_INT,&mpi_vect2);
        MPI_Type_commit(&mpi_vect2);

        // MPI_Recv(a_loc,1,mpi_vect,0,34,MPI_COMM_WORLD,&status); //doesn't work - Reason: mpi_vect has stride = 4. We need stride = 2 to populate 3X2 array. So, define new MPI_Datatype
        
        MPI_Recv(a_loc,1,mpi_vect2,0,34,MPI_COMM_WORLD,&status); //works
        // MPI_Recv(a_loc,6,MPI_INT,0,34,MPI_COMM_WORLD,&status); //works
        for(int i=0;i<3;i++)
        {
            for(int j=0;j<2;j++)
            {
                cout << a_loc[i][j] << " ";
            }
            cout << endl;
        }
    }


    

    MPI_Finalize();

}