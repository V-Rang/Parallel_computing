#include<iostream>
#include<omp.h>
#include<climits>
#include<bitset>
#include<random> 


using namespace std;


int main()
{
    omp_set_num_threads(5);
    int tid;
    int i,j;

    float cur_max = -std::numeric_limits<float>::max();
    // cout << cur_max << endl;
    int n = (int)10;
    double a[n];


    const double minval = 1.0;
    const double maxval = 10.0;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double>dist(minval,maxval);


    for(i=0;i<n;i++)
    {
        a[i] = dist(gen);
    }

    for(i=0;i<n;i++)
    {
        cout << a[i]<<" ";
    }
    cout << endl;

    //sequentially finding the maximum value in array a:
    for(i=0;i<n;i++)
    {
        if(a[i]>cur_max)
        {
            cur_max = a[i];
        }
    }

    printf("The maximum value in the array is %f\n",cur_max);

    // //using locks:
    cur_max = -std::numeric_limits<float>::max(); //reinitializing
    //M-1
    // void omp_init_lock(omp_lock_t *test_lock);

    //M-2
    omp_lock_t(test_lock);
    omp_init_lock(&test_lock);

    #pragma omp parallel for
    for(i=0;i<n;i++)
    {
        if(a[i]>cur_max)
        {
            omp_set_lock(&test_lock);
            // omp_set_lock(&test_lock); // causes deadlock because lock won't be unset till later => synchronization event that will never happen
            if(a[i] > cur_max)
            {
                cur_max = a[i];
            }
            omp_unset_lock(&test_lock);
        }
    }
    omp_destroy_lock(&test_lock);
    printf("The maximum value in the array using omp locks is %f\n",cur_max);





    return 0;
}