/*
Trying to scatter 2X2 blocks from a 4X4 matrix.
Cases:
1. One block of 2X2.
2. Two blocks of 2X2 -> received and reshaped for:
    a. 8 length array
    b. 2 X 4 array
    c. 4 X2 array

The second block for case 2 starts at the next memory location to the last element in the first block.
So, if array looks like: 
    [[1,2,3,4],
    [5,6,7,8],
    [9,10,11,12],
    [13,14,15,16]]

and the first block of 2X2 starts at a[0][0], the first block will be: [[1,2],
                                                                        [5,6]]
and the second block will be: [[7,8],
                               [11,12]]

The above 8 elements will be sent and reshaped according to the receiving array.
*/

#include<iostream>
#include<mpi.h>

using namespace std;

void build_cust_d(int *row, int *col, float *values, MPI_Datatype *mpi_mesg)
{
    int blocklengths[3];
    MPI_Datatype typelists[3];
    MPI_Aint displacements[3];

    blocklengths[0] = blocklengths[1] = blocklengths[2] = 2;

    typelists[0] = typelists[1] = MPI_INT;
    typelists[2] = MPI_FLOAT;

    MPI_Aint address;
    MPI_Aint start_address;

    MPI_Get_address(row,&start_address);

    MPI_Get_address(col,&address);

    displacements[0] = 0;
    displacements[1] = address - start_address;

    MPI_Get_address(values,&address);
    displacements[2] = address - start_address;

    MPI_Type_create_struct(3,blocklengths,displacements,typelists,mpi_mesg);
    MPI_Type_commit(mpi_mesg);
}

int main(int argc, char **argv)
{
    int my_node,total_nodes;
    MPI_Status status;
    MPI_Init(&argc,&argv);
    MPI_Comm_rank(MPI_COMM_WORLD,&my_node);
    MPI_Comm_size(MPI_COMM_WORLD,&total_nodes);

    int a[4][4] = {{1,2,3,4},{5,6,7,8},{9,10,11,12},{13,14,15,16}};
    int a_loc[2][2]; //2 X2  to receive one block of 2 x 2
    int a_loc2[8]; // 8 length array to receive two blocks of 2 X2
    int a_loc3[2][4]; // 2X4 array to recieve two blocks of 2X2, the 8 values are reshaped to fill in the 2X4 array.
    int a_loc4[4][2]; // 4X2 array to recieve two blocks of 2X2, the 8 values are reshaped to fill in the 4X2 array.
    int blocklengths[2];
    blocklengths[0] = blocklengths[1] = 2;
    int displacements[2];
    displacements[0] = 0;
    displacements[1] = 4;
    MPI_Datatype mpi_block;

    MPI_Type_indexed(2,blocklengths,displacements,MPI_INT,&mpi_block); //define a block of 2 X 2
    
    MPI_Type_commit(&mpi_block);

    if(my_node == 0)
    {
        a[0][0] = 2;
        a[0][1] = 4;
        a[1][0] = 6;
        a[1][1] = 8;

        // MPI_Send(&(a[0][0]),1,mpi_block,1,34,MPI_COMM_WORLD); //send one block of 2 X 2
        MPI_Send(&(a[0][0]),2,mpi_block,1,34,MPI_COMM_WORLD); //send two blocks of 2 X 2
    
    }
    
    if(my_node == 1)
    {
        // MPI_Recv(a_loc,4,MPI_INT,0,34,MPI_COMM_WORLD,&status); //to recive the 4 values sent in the one block
        // MPI_Recv(a_loc2,8,MPI_INT,0,34,MPI_COMM_WORLD,&status); //to receive the 8 values sent in the two blocks.
        MPI_Recv(a_loc4,8,MPI_INT,0,34,MPI_COMM_WORLD,&status);

        //print values recieved in the one block of 2X2
        // for(int i=0;i<2;i++)
        // {
        //     for(int j=0;j<2;j++)
        //     {
        //         cout << a_loc[i][j] << " ";
        //     }
        //     cout << endl;
        // }
        
        //print values recieved in the two blocks of 2 X 2 (total 8 values)
        // for(int i=0;i<8;i++)
        // {
        //     cout << a_loc2[i] << " "; 
        // }
        // cout << endl;

      //print values recieved in the two blocks of 2 X 2 (total 8 values)
    //   for(int i=0;i<2;i++)
    //   {
    //     for(int j=0;j<4;j++)
    //     {
    //         cout << a_loc3[i][j] << " ";
    //     }
    //     cout << endl;
    //   }

      //print values recieved in the two blocks of 2 X 2 (total 8 values)
      for(int i=0;i<4;i++)
      {
        for(int j=0;j<2;j++)
        {
            cout << a_loc4[i][j] << " ";
        }
        cout << endl;
      }


    }
    
   


    

    MPI_Finalize();

}