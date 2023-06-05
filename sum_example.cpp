#include<iostream>
#include<mpi.h>

using namespace std;

int main(int argc, char **argv)
{
    int my_node,total_nodes;
    int sum,startval,endval,accum;
    MPI_Status status;
    MPI_Init(&argc,&argv);

    MPI_Comm_rank(MPI_COMM_WORLD,&my_node);
    MPI_Comm_size(MPI_COMM_WORLD,&total_nodes);

    sum = 0;
    startval = my_node*1000/total_nodes + 1;
    endval = (my_node + 1)*1000/total_nodes;

    for(int i=startval;i<=endval;i++)
    {
        sum = sum + i;
    }

    if(my_node != 0)
    {
        MPI_Send(&sum,1,MPI_INT,0,1,MPI_COMM_WORLD);
    }
    else
    {
        for(int j=1;j<total_nodes;j++)
        {
            MPI_Recv(&accum,1,MPI_INT,j,1,MPI_COMM_WORLD,&status);
            sum = sum + accum;
        }
    }

    if(my_node == 0)
    {
        cout << "The sum from 1 to 1000 is: " << sum << endl;
    }

    MPI_Finalize();

}