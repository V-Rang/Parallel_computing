/*
Problem Statement: 
1. Have process 0 query user for number of elements over which to sum.
2. From process 0, distribute to all processes the number of elements to sum
(using sends and receives) and appropriately calculate the interval over which
each process is to sum.
3.  Accomplish the summing.
4.  After creating the final answer on process zero, print the result.


Tu run program:
mpiexec -n 1 ./mpi_sum_elements.exe 1000 : -n 4 ./mpi_sum_elements.exe 50

The above command will calculate sum of the first 1000 natural numbers using 5 processes.
1000 is sent to process 0 which then sends this value to the other 4 processes.
50 is a  garbage value that is initially sent to the other 4 processes.
*/

#include<iostream>
#include<mpi.h>
using namespace std;

int main(int argc, char **argv)
{

    int my_node,total_nodes;
    MPI_Status status;
    double sum,accum;
    MPI_Init(&argc,&argv);
    MPI_Comm_rank(MPI_COMM_WORLD,&my_node);
    MPI_Comm_size(MPI_COMM_WORLD,&total_nodes);
    
    int no_ele = stoi(argv[1]);
    sum = 0.0;

    // cout << "Process " << my_node << " has the value " << no_ele << endl;

    //using Bcast
    MPI_Bcast(&no_ele,1,MPI_INT,0,MPI_COMM_WORLD);
    // using Send and Recv
    if(my_node == 0)
    {
        for(int j=1;j<total_nodes;j++)
        {
            MPI_Send(&no_ele,1,MPI_INT,j,1,MPI_COMM_WORLD);
        }
    }
    else
    {
        MPI_Recv(&no_ele,1,MPI_INT,0,1,MPI_COMM_WORLD,&status);
    }

    // cout << "Process " << my_node << " has the value " << no_ele << endl;

    int start_index = my_node*no_ele/total_nodes + 1;
    int end_index = (my_node+1)*no_ele/total_nodes;

    for(int j=start_index;j<=end_index;j++)
    {
        sum += j;
    }

    if(my_node == 0)
    {
        for(int j=1;j<total_nodes;j++)
        {
            MPI_Recv(&accum,1,MPI_DOUBLE,j,1,MPI_COMM_WORLD,&status);
            sum += accum;
        }
    }
    else
    {
        MPI_Send(&sum,1,MPI_DOUBLE,0,1,MPI_COMM_WORLD);
    }

    if(my_node == 0)
    {
        printf("Sum of first %d natural numbers = %lf\n",no_ele,sum);
    }

    MPI_Finalize();

}
