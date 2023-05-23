/*
Sum of even and odd index element of an array.
*/

#include<iostream>
#include<omp.h>
#include<random>

using namespace std;

int main()
{
    const double minval = 1.0;
    const double maxval = 100.0;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double>dist(minval,maxval);

    int n = (int)1e6; //number of elements
    double a[n] = {0}; // array
    double res[2] = {0}; // array that contains result of serial execution 0-> even index, 1-> odd index

    int i,j;
    for(i=0;i<n;i++)
    {
        a[i] = dist(gen); 
        // a[i] += i;
    }

    // for(i=0;i<n;i++)
    // {
    //     // a[i] = dist(gen);
    //     cout << a[i] << " "; 
    // }
    // cout << endl;


    for(i=0;i<n-1;i += 2)
    {
        res[0] += a[i]; //even index elements
        res[1] += a[i+1]; //odd index elements
    }

    if(n%2 != 0)
    {
        res[0] += a[n-1];
    }
    cout << "Serial: Sum of even index elements = " << res[0]  << endl;
    cout << "Serial: Sum of odd index elements = " << res[1]  << endl;
    
    
    //in parallel
    int num_threads = 4;
    omp_set_num_threads(num_threads);

    // double local_s[2][omp_get_num_threads()]; //this messes up the code in intializing the values of local_s. I guess because of dynamic threads.
    double local_s[2][num_threads] = {0};

    // for(i=0;i<2;i++)
    // {
    //     for(j=0;j<5;j++)
    //     {       
    //         local_s[i][j] = i+2*j;
    //         printf("ivalue = %d, j value = %d, value to be inserted = %d\n",i,j,i+2*j);
    //     }
    // }

    // local_s[0][3] = 6;

    // for(i=0;i<2;i++)
    // {
    //     for(j=0;j<5;j++)
    //     {
    //         cout << local_s[i][j] << " ";
    //     }
    //     cout << endl;
    // }

    // cout << local_s[0][3] << endl;
    
    int tid;
    int index;
    double eve_sum = 0;
    double odd_sum = 0;



    // Method - 1 Using shared array, slower because of false sharing. 
    #pragma omp parallel private(tid,i) shared(eve_sum,odd_sum,local_s,a,n) default(none) // private or non-private i makes no difference
    {
        tid = omp_get_thread_num();
        #pragma omp for schedule(static) private(i) // private(tid) messes up
        for(i=0;i<n;i++) 
        {
            // printf("Thread %d executing for iteration index %d and value = %f\n",tid,i,a[i]);
            // printf("Row %d, column %d and iteration index = %d, value = %f\n",i%2,tid,i,a[i]);
            // printf("Thread: %d has iteration value: %d\n",tid,i);
            local_s[i%2][tid] += a[i];
        }

        #pragma omp atomic
        eve_sum += local_s[0][tid];
        
        #pragma omp atomic
        odd_sum += local_s[1][tid];
        
    }

    cout << "Method1: Parallel sum of even index elements = " << eve_sum << endl;
    cout << "Method1: Parallel sum of odd index elements = " << odd_sum  << endl;

    //Method - 2 Using distinct cache lines
    double local_s2[2];
    double eve_sum2 = 0;
    double odd_sum2 = 0;

    #pragma omp parallel private(local_s2) shared(a,n,eve_sum2,odd_sum2) default(none)
    {
        local_s2[0] = 0;
        local_s2[1] = 0;
        #pragma omp for private(i)
        for(i=0;i<n;i++)
        {
            local_s2[i%2] += a[i];
        }
        #pragma omp atomic
        eve_sum2 += local_s2[0];
        
        #pragma omp atomic
        odd_sum2 += local_s2[1];
    }

    cout << "Method2: Parallel sum of even index elements = " << eve_sum2 << endl;
    cout << "Method2: Parallel sum of odd index elements = " << odd_sum2  << endl;

    return 0;
}