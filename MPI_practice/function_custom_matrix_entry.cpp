/*
Function that builds a custom derived data type that refers to a  matrix entry (row,col,value). This is then
modified by proc0 which then Bcasts it to the other processes. 
*/

#include<iostream>
#include<mpi.h>

using namespace std;


void build_custom_matrixentry(int *rows, int *cols, float *values, MPI_Datatype *mpi_new_t)
{
    int block_lengths[3];
    MPI_Aint displacements[3];
    MPI_Datatype typelists[3];

    block_lengths[0] = block_lengths[1] = block_lengths[2] = 1;

    typelists[0] = MPI_INT;
    typelists[1] = MPI_INT;
    typelists[2] = MPI_FLOAT;

    MPI_Aint address;
    MPI_Aint start_address;

    displacements[0] = 0;

    MPI_Get_address(rows,&start_address);
    MPI_Get_address(cols,&address);
    
    displacements[1] = address - start_address;

    MPI_Get_address(values,&address);
    
    displacements[2] = address - start_address;

    MPI_Type_create_struct(3,block_lengths,displacements,typelists,mpi_new_t);

}

//following two structs not needed for MPI code
struct matrix_entry
{
    int row;
    int col;
    float value;

    matrix_entry() {}
    matrix_entry(int row_ind, int col_ind ,float val): row(row_ind),col(col_ind),value(val) {}
};

struct matrix
{
    matrix_entry* mat;

    matrix(): mat(NULL) {}

};



int main(int argc, char **argv)
{
    int my_node,total_nodes;
   
    MPI_Status status;
    MPI_Init(&argc,&argv);
    MPI_Comm_rank(MPI_COMM_WORLD,&my_node);
    MPI_Comm_size(MPI_COMM_WORLD,&total_nodes);

    int *row_index, *col_index;
    float *entry_vals;

    row_index = new int[3]{0,1,2};
    col_index = new int[3]{3,4,5};
    entry_vals = new float[3]{6.7,7.8,8.9};

    MPI_Datatype new_mpi_struct;

    if(my_node == 0)
    {
        entry_vals[1] = 13.56;
    }
    
    build_custom_matrixentry(row_index,col_index,entry_vals,&new_mpi_struct);
    MPI_Type_commit(&new_mpi_struct);

    MPI_Bcast(row_index,1,new_mpi_struct,0,MPI_COMM_WORLD);
    
    if(my_node == 1)
    {
        for(int i=0;i<3;i++)
        {
            cout << row_index[i] << " " << col_index[i] << " " << entry_vals[i] << endl;
        }
    }


    MPI_Finalize();

}