/*
Trapezoidal rule timing using serial, OpenMP and OpenMPI.
Have to comment out the mpi portion of code (including mpi.h) when runing serial and OpenMP implementation of code.
*/

#include<iostream>
#include<math.h>
#include<omp.h>
// #include<mpi.h>

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
    // return pow(x,2); //example function
    return x;
}


double trap_integral_serial(double a, double b, int n, double (*func)(double x) )
{

    double integral = (func(a) + func(b))/2;
    double h = (b-a)/n;
    for(int i=1;i<=n-1;i++)
    {
        integral += func(a + i*h);
    }
    return integral * h;
}

double trap_integral_parallel_mp(double a, double b, int n, double (*func)(double x) )
{
    int i;
    double integral = (func(a) + func(b))/2;
    double h = (b-a)/n;
    #pragma omp parallel for private(i) shared(func,n,a,h) default(none) reduction(+:integral)
    for(i=1;i<=n-1;i++)
    {
        integral += func(a + i*h);
    }

    return integral * h;
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

    int timing_iterations = (int)2000; //number of times to run serial, openMP or openMPI code and then divide by to calculate average time.    
    float ellapsed_time;
    Cputimer timer;

    // M-1 Serial calculation
    double serial_ans;
    timer.Start();
    for(int i=0;i<timing_iterations;i++)
    {
        serial_ans = trap_integral_serial(1,(int)1e4,(int)900000,&func_eval);
    }
    timer.Stop();
    ellapsed_time = timer.EllapsedMicros()/timing_iterations;
    printf("Answer for serial calculation = %lf and time taken = %lf\n",serial_ans,ellapsed_time);

    //M-2 OpenMP Calculation
    double openmp_ans;
    timer.Start();
    for(int i=0;i<timing_iterations;i++)
    {
        openmp_ans = trap_integral_parallel_mp(1,(int)1e4,(int)900000,&func_eval);
    }
    timer.Stop();
     ellapsed_time = timer.EllapsedMicros()/timing_iterations;
    printf("Answer for openMp calculation = %lf and time taken = %lf\n",openmp_ans,ellapsed_time);


    //OpenMPI calculation
    /*
    (works even if n-1 is not exactly divisible by p) 
    Why (n-1):
    Trapezoidal formulated as: a,x1,x2,...x(n-1),b.
    To calculate: [f(a)/2 + f(b)/2 + f(x1) + ... + f(x(n-1) )]*h.
    Process 0 calculates (f(a) + f(b))/2.
    The (n-1) sums are distributed to all processes such that: 
    n-1 = pq + r. (q = quotient, r=remainder). 
        = r(q+1) + (p-r)(q) i.e. the first r processes process the q+1 "iterations"/"indices" each  while
        the remaining (p-r) processes process q "iterations"/"indices" each.
    */

   
    //M-3 OpenMPI     
    // double  a,b;
    // int n;
    // a = 1,
    // b = (int)1e4;
    // n = 900000;
    
    // int my_node,total_nodes;
    // MPI_Status status;
    // MPI_Init(&argc,&argv);
    // MPI_Comm_rank(MPI_COMM_WORLD,&my_node);
    // MPI_Comm_size(MPI_COMM_WORLD,&total_nodes);
    // double h = (b-a)/n;
    // double sum  = 0.0;
    // double accum;

    // int q = (n-1)/total_nodes;
    // int r = (n-1) % total_nodes;
    // int start_index,end_index;
    // if(my_node < r)
    // {
    //     start_index = my_node*q + my_node; 
    //     end_index = start_index + q;
    // }
    // else
    // {
    //     start_index  = my_node*q + r;
    //     end_index = start_index + q-1;
    // }
    
    // //uncomment following line to see distribution of sum of x1...x(n-1) to p processes.
    // // printf("Process %d has start_index = %d and end_index = %d\n",my_node,start_index,end_index);
    

    // for(int i=0;i<timing_iterations;i++)
    // {
    //     timer.Start();       
    //     sum = trap_integral_mpi(a,start_index,end_index,h);
    //     if(my_node != 0)
    //     {
    //         MPI_Send(&sum,1,MPI_DOUBLE,0,1,MPI_COMM_WORLD);
    //     }
    //     else
    //     {
    //         for(int j=1;j<total_nodes;j++)
    //         {
    //             MPI_Recv(&accum,1,MPI_DOUBLE,j,1,MPI_COMM_WORLD,&status);
    //             sum += accum;
    //         }
    //         sum += (func_eval(a) + func_eval(b))/2;
    //         sum *= h;
    //     }
    // }
    // if(my_node == 0)
    // {
    //     timer.Stop();
    //     ellapsed_time = timer.EllapsedMicros()/timing_iterations;
    //     printf("Answer for openMpI calculation = %lf and time taken = %lf\n",sum,ellapsed_time);
    // }

    // MPI_Finalize();
    


    return 0;
}