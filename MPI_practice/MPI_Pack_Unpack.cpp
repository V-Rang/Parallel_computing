/*
MPI_Pack and MPI_Unpack
buf_size: Use MPI_Pack_size to get maximum size of bytes needed in buffer.
Buffer type: char , don't know why. Works with int type too.
*/

#include<iostream>
#include<mpi.h>

using namespace std;

int main(int argc, char **argv)
{
        
    int a[4][4] = {{1,0,1,0},
                   {0,0,0,0},
                   {0,0,3,3},
                   {4,4,4,4}};

    int my_node,total_nodes;
    MPI_Status status;
    MPI_Init(&argc,&argv);
    
    MPI_Comm_rank(MPI_COMM_WORLD,&my_node);
    MPI_Comm_size(MPI_COMM_WORLD,&total_nodes);
    int buf_size = 40; //1int(4)+ 1int(4) + 4col_inds(4X4) + 4entries(4X4) = 40
    char buffer[buf_size];
    int row_number;
    int nnz;
    int *col_inds;
    float *entries;
    int position;
    //sending row 3
    if(my_node == 0)
    {
        position = 0;
        row_number = 3;
        nnz = 4;
        col_inds = new int[nnz] {0,1,2,3};
        entries = new float[nnz] {4,4,4,4};
        
        MPI_Pack(&row_number,1,MPI_INT,buffer,buf_size,&position,MPI_COMM_WORLD);        
        MPI_Pack(&nnz,1,MPI_INT,buffer,buf_size,&position,MPI_COMM_WORLD);
        MPI_Pack(col_inds,nnz,MPI_INT,buffer,buf_size,&position,MPI_COMM_WORLD);
        MPI_Pack(entries,nnz,MPI_FLOAT,buffer,buf_size,&position,MPI_COMM_WORLD);
        
        // int max_size;
        // MPI_Pack_size(1,MPI_FLOAT,MPI_COMM_WORLD,&max_size); //max_size stores size of buffer needed. Can do for all Pack calls to get maxuimum buffer size needed
        // cout << max_size;

        MPI_Send(buffer,position,MPI_PACKED,1,34,MPI_COMM_WORLD);
    }
    else
    {
        MPI_Recv(buffer,buf_size,MPI_PACKED,0,34,MPI_COMM_WORLD,&status);
        position = 0;
        MPI_Unpack(buffer,buf_size,&position,&row_number,1,MPI_INT,MPI_COMM_WORLD);
        MPI_Unpack(buffer,buf_size,&position,&nnz,1,MPI_INT,MPI_COMM_WORLD);
        
        col_inds = new int[nnz];
        entries = new float[nnz];

        MPI_Unpack(buffer,buf_size,&position,col_inds,nnz,MPI_INT,MPI_COMM_WORLD);
        MPI_Unpack(buffer,buf_size,&position,entries,nnz,MPI_FLOAT,MPI_COMM_WORLD);

        printf("Row number = %d\n",row_number);
        printf("Number of nnz = %d\n",nnz);
        printf("col_inds and entries =:\n");
        for(int i=0;i<nnz;i++)
        {
            cout << col_inds[i] << " " << entries[i] << endl;
        }

    }


    
       

    MPI_Finalize();


}