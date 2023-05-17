/*
Finding the maximum element in an array using:
1. Serial execution
2. Critical sections
3. Atomic directive
4. Runtime library lock routine
*/

#include<iostream>
#include<omp.h>
#include<random>

using namespace std;

int main()
{
    omp_set_num_threads(5);
    int tid;
    int i,j;

    float cur_max = -std::numeric_limits<float>::max();
    // cout << cur_max << endl;
    int n = (int)1e5;
    double a[n];


    const double minval = 1.0;
    const double maxval = 100.0;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double>dist(minval,maxval);


    for(i=0;i<n;i++)
    {
        a[i] = dist(gen);
    }

    // for(i=0;i<n;i++)
    // {
    //     cout << a[i]<<" ";
    // }
    // cout << endl;

    cur_max = -std::numeric_limits<float>::max(); //reinitializing

    //find max element serially
    for(i=0;i<n;i++)
    {
        if(a[i] > cur_max)
        {
            cur_max = a[i];
        }
    }
    
    cout <<"Maximum using serial execution: " <<cur_max << endl;


    // using critical sections
    cur_max = -std::numeric_limits<float>::max(); //reinitializing
    #pragma omp parallel for
    for(i=0;i<n;i++)
    {
        if(a[i] > cur_max)
        #pragma omp critical
        {
            if(a[i] > cur_max)
            {
                cur_max = a[i];
            }
        }
    }

    cout << "Maximum using critical sections: " << cur_max << endl;

    // //using atomic directive - not working
    // cur_max = -std::numeric_limits<float>::max(); //reinitializing
    // // float cur_max2;
    // #pragma omp parallel for 
    // for(i=0;i<n;i++)
    // {

    //     if(a[i] > cur_max)
    //     {
    //         // #pragma omp single //need to figure out how to get exclusive access here so cur_max doesn't change from line above
    //         // cur_max2 = cur_max;
    //         if(a[i] > cur_max)
    //         {
    //             #pragma omp atomic
    //             cur_max -= cur_max;

    //             #pragma omp atomic
    //             cur_max += a[i];
    //         }
    //         // cur_max = cur_max2;
    //     }

    // }
    // cout << "Maximum using atomic directive: " << cur_max << endl;

    // Maximum using atomic directive - works
    cur_max = -std::numeric_limits<float>::max(); //reinitializing
    // float cur_max2 = 0.0;
    #pragma omp parallel for
    for(i=0;i<n;i++)
    {
        if(a[i] > cur_max)
        {
            #pragma omp acquire flush cur_max // code works even without three flush statements

            if(a[i] > cur_max)
            {
                #pragma omp acquire flush cur_max

                #pragma omp atomic write
                cur_max = a[i];

                // #pragma omp atomic
                // cur_max -= cur_max;

                // #pragma omp atomic
                // cur_max += a[i];
                
                #pragma omp release flush cur_max
            }
        }

    }
    cout << "Maximum using atomic directive: "<<cur_max << endl;



    //using runtime library lock routines
    cur_max = -std::numeric_limits<float>::max(); //reinitializing
    omp_lock_t(test_lock);
    omp_init_lock(&test_lock);

    #pragma omp parallel for
    for(i=0;i<n;i++)
    {
        if(a[i]>cur_max)
        {
            omp_set_lock(&test_lock);
            if(a[i] > cur_max)
            {
                cur_max = a[i];
            }
            omp_unset_lock(&test_lock);
        }
    }
    omp_destroy_lock(&test_lock);

    cout << "Maximum using runtime library lock routine: " << cur_max << endl;

    return 0;
}

// #pragma omp parallel for
//     for(i=0;i<n;i++)
//     {
//         if(a[i]>cur_max)
//         {
//             #pragma omp atomic
//             cur_max -= cur_max;
//             #pragma omp atomic
//             cur_max += a[i]; 

//         }
//     }