/*
Function that builds a custom derived data type that refers to a  matrix entry (row,col,value). This is then
modified by proc0 which then Bcasts it to the other processes. 
*/

#include<iostream>
#include<mpi.h>

using namespace std;

void build_cust_d(int *row_index, int *col_index, float *vals, MPI_Datatype* mpi_mesg)
{
    int blocklengths[3];
    MPI_Aint displacements[3];
    MPI_Datatype typelists[3];

    blocklengths[0] = 3;
    blocklengths[1] = 3;
    blocklengths[2] = 3;

    typelists[0] = MPI_INT;
    typelists[1] = MPI_INT;
    typelists[2] = MPI_FLOAT;

    MPI_Aint address;
    MPI_Aint start_address;

    displacements[0] = 0;

    MPI_Get_address(row_index,&start_address);
    MPI_Get_address(col_index,&address);

    displacements[1] = address - start_address;

    MPI_Get_address(vals,&address);

    displacements[2] = address - start_address;

    MPI_Type_create_struct(3,blocklengths,displacements,typelists,mpi_mesg);

    MPI_Type_commit(mpi_mesg);
    
}

int main(int argc, char **argv)
{
    int *row;
    int *col;
    float *entry_vals;

    row = new int[3]{10,20,30};
    col = new int[3]{40,50,60};
    entry_vals = new float[3]{12,23,34};

    int my_node,total_nodes;
    MPI_Init(&argc,&argv);
    MPI_Comm_rank(MPI_COMM_WORLD,&my_node);
    MPI_Comm_size(MPI_COMM_WORLD,&total_nodes);    

    MPI_Datatype new_mpi_t;

    if(my_node == 0)
    {
        *row = 1;
        *(row+1) = 2;
        *(row+2) = 3;

        *col = 4;
        *(col+1) = 5;
        *(col+2) = 6;

        *entry_vals = 1.2;
        *(entry_vals+1) = 2.3;
        *(entry_vals+2) = 3.4;
    }
    
    build_cust_d(row,col,entry_vals,&new_mpi_t);

    MPI_Bcast(row,1,new_mpi_t,0,MPI_COMM_WORLD);

    if(my_node == 1)
    {
        for(int i=0;i<3;i++)
        {
            cout << row[i] << " " << col[i] << " " << entry_vals[i] << endl;
        }
    }


    MPI_Finalize();

}
