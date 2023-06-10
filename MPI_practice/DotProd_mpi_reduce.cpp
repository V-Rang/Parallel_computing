/*
Dot product using MPI_Reduce

*/


#include<iostream>
#include<vector>
#include<mpi.h>



using namespace std;

int main(int argc, char **argv)
{
    vector<double>a = {1,2,3,4,5};
    vector<double>b = {5,4,3,2,1};
    vector<double>c;

    int my_node,total_nodes;
    MPI_Init(&argc,&argv);
    MPI_Comm_rank(MPI_COMM_WORLD,&my_node);
    MPI_Comm_size(MPI_COMM_WORLD,&total_nodes);

    int n = a.size();
    
    int q = n/total_nodes;
    int r = n % total_nodes;
    int start_index,end_index;
    if(my_node < r)
    {
        start_index = my_node*q + my_node; 
        end_index = start_index + q;
    }
    else
    {
        start_index  = my_node*q + r;
        end_index = start_index + q-1;
    }

    double loc_sum = 0.0;
    double glob_sum = 0.0;

    for(int i=start_index;i<=end_index;i++)
    {
        loc_sum += a[i]*b[i];
    }

    //To have only process 0 have the end result while other procs have result = 0.
    MPI_Reduce(&loc_sum,&glob_sum,1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);

    //If want all procs to have end result which can then be used for future calculations:
    // MPI_Allreduce(&loc_sum,&glob_sum,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);


    if(my_node == 0)
    {
        printf("Dot product of vectors a and b = %lf\n",glob_sum);
    }

   

    MPI_Finalize();


}
