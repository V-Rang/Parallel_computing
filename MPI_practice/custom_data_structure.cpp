/*
custom data structure to communicate multiple variable values in MPI using MPI_Type_create_struct
*/

#include<iostream>
#include<mpi.h>

using namespace std;

void build_cust_d(int *a, char *b, float *c, MPI_Datatype* mpi_mesg)
{
    int blocklengths[3];
    MPI_Aint displacements[3];
    MPI_Datatype typelists[3];

    blocklengths[0] = 3;
    blocklengths[1] = 2;
    blocklengths[2] = 4;

    typelists[0] = MPI_INT;
    typelists[1] = MPI_CHAR;
    typelists[2] = MPI_FLOAT;

    MPI_Aint address;
    MPI_Aint start_address;

    displacements[0] = 0;

    MPI_Get_address(a,&start_address);
    MPI_Get_address(b,&address);

    displacements[1] = address - start_address;

    MPI_Get_address(c,&address);

    displacements[2] = address - start_address;

    MPI_Type_create_struct(3,blocklengths,displacements,typelists,mpi_mesg);
    MPI_Type_commit(mpi_mesg);
    
}

int main(int argc, char **argv)
{
    int my_node,total_nodes;
    MPI_Init(&argc,&argv);
    MPI_Comm_rank(MPI_COMM_WORLD,&my_node);
    MPI_Comm_size(MPI_COMM_WORLD,&total_nodes);
    int *a = new int[3]{1,2,3};
    char *b = new char[2]{'a','b'};
    float *c = new float[4]{1.2,2.3,3.4,4.5};

    if(my_node == 0)
    {
        b[1] = 'c';
        c[3] = 5.6;
    }

    MPI_Datatype new_mpi_t;
    
    build_cust_d(a,b,c,&new_mpi_t);
    MPI_Bcast(a,1,new_mpi_t,0,MPI_COMM_WORLD);


    if(my_node == 1)
    {
        cout << b[1] << " " << c[3] << endl;
    }


    MPI_Finalize();

}
