#include<omp.h>
#include<stdio.h>
#include<time.h>

static long num_steps = 1000000000;
double step;



void main()
{
    //Serial pi
    clock_t start,end;
    int i;
    step = (1-0)/(double)num_steps;
    double x,pi,sum=0.0;

    start = clock();
    for(i=0;i<num_steps;i++)
    {
        sum +=  4/(1 + ( i/(double)num_steps)*( i/(double)num_steps));
    }
    pi = step*sum;
    end = clock();
    // printf("pi value = %f\n",pi);
    printf("pi value = %f and time taken = %f\n",pi,((float)end-(float)start)/CLOCKS_PER_SEC);


    //Method 1
    sum = 0.0;
    start = clock();
    #pragma omp parallel num_threads(4) reduction(+:sum) shared(num_steps) private(i) default(none)
    {
        #pragma omp for
        for(i=0;i<num_steps;i++)
        {
            sum +=  4.0/(1.0 +  (i/(double)num_steps) * (i/(double)num_steps));
        }
    }
    pi = step*sum;
    end = clock();
    // printf("pi value = %f\n",pi);
    printf("pi value = %f and time taken = %f\n",pi, ( (float)end-(float)start)/CLOCKS_PER_SEC);

    //Method 2
    sum = 0.0;
    start = clock();
    #pragma omp parallel for num_threads(4) reduction(+:sum) shared(num_steps) private(i) default(none)
    for(i=0;i<num_steps;i++)
    {
        sum +=  4.0/(1.0 +  (i/(double)num_steps) * (i/(double)num_steps));
    }
    
    pi = step*sum;
    end = clock();
    printf("pi value = %f and time taken = %f\n",pi, ( (float)end-(float)start)/CLOCKS_PER_SEC);
    // printf("pi value = %f\n",pi);


}
