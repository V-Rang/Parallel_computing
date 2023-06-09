// /*
// Trapezoidal using reduction
// */

#include<iostream>
#include<omp.h>
#include<mpi.h>

using namespace std;


struct Cputimer
{

    double start;
    double stop;

    void Start()
    {
        start = omp_get_wtime(); 
    }

    void Stop()
    {
        stop = omp_get_wtime();
    }

    float EllapsedMicros()
    {
        return (stop-start)*1e6;
    }
};



double func_eval(double x)
{
    return x;
}

double trap_integral_mpi(double a, int start_index, int end_index , double h)
{
    double sum_loc = 0.0;
    for(int i=start_index+1;i<=end_index+1;i++) // trapezoidal rule: a,x1,x2,x3,...x(n-1),b. x1 = a + 1*i*h. So have to start from start_index + 1 and end at end_index + 1.
    {
        sum_loc += a + i*h;
    }
    return sum_loc;
}


int main(int argc, char **argv)
{
    Cputimer timer;
    int timing_iterations = 2000;
    float ellapsed_time;

    // int my_node,total_nodes;
    // MPI_Init(&argc,&argv);
    // MPI_Comm_rank(MPI_COMM_WORLD,&my_node);
    // MPI_Comm_size(MPI_COMM_WORLD,&total_nodes);
    // printf("Hello from process %d out of process %d\n",my_node,total_nodes);

     double  a,b;
    int n;
    a = 1,
    b = (int)1e4;
    n = 900000;
    
    int my_node,total_nodes;
    MPI_Status status;
    MPI_Init(&argc,&argv);
    MPI_Comm_rank(MPI_COMM_WORLD,&my_node);
    MPI_Comm_size(MPI_COMM_WORLD,&total_nodes);
    double h = (b-a)/n;
    double sum  = 0.0;
    double accum;

    int q = (n-1)/total_nodes;
    int r = (n-1) % total_nodes;
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
    
    //uncomment following line to see distribution of sum of x1...x(n-1) to p processes.
    // printf("Process %d has start_index = %d and end_index = %d\n",my_node,start_index,end_index);
    
    double integral = 0; //need to define a new variable. "Sum" of all the procs cannnot be stored in the "Sum" of proc 0 - "aliasing of arguments" problem.   
    for(int i=0;i<timing_iterations;i++)
    {
        timer.Start();       

        sum = trap_integral_mpi(a,start_index,end_index,h);
       
        MPI_Reduce(&sum,&integral,1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
        
        //without reduction
        // if(my_node != 0)
        // {
        //     MPI_Send(&sum,1,MPI_DOUBLE,0,1,MPI_COMM_WORLD);
        // }
        // else
        // {
        //     for(int j=1;j<total_nodes;j++)
        //     {
        //         MPI_Recv(&accum,1,MPI_DOUBLE,j,1,MPI_COMM_WORLD,&status);
        //         sum += accum;
        //     }
        //     sum += (func_eval(a) + func_eval(b))/2;
        //     sum *= h;
        // }
    }
    if(my_node == 0)
    {
        integral += (func_eval(a) + func_eval(b))/2;
        integral *= h;
        timer.Stop();

        ellapsed_time = timer.EllapsedMicros()/timing_iterations;
        printf("Answer for openMpI calculation = %lf and time taken = %lf\n",integral,ellapsed_time);
    }


    MPI_Finalize();
}

//***********************************************************
// Serial code that works
// #include<iostream>

// using namespace std;


// int main()
// {
//     double a,b;
//     int n;
//     a = 1;
//     b = 10;
//     n = 9;

//     double h = (b-a)/n;

//     double sum = (func_eval(a) + func_eval(b))/2;

//     for(int i=1;i<=n-1;i++)
//     {
//         sum += func_eval(a+i*h);
//     }

//     cout << sum << endl;

//     return 0;
// }