#include<omp.h>
#include<stdio.h>
#include<time.h>

static long num_steps = 100000;
double step;

void main()
{
    //Serial pi
    int i;
    step = (1-0)/(double)num_steps;
    double x,pi,sum=0.0;

    for(i=0;i<num_steps;i++)
    {
        sum +=  4/(1 + ( i/(double)num_steps)*( i/(double)num_steps));
    }
    pi = step*sum;
    printf("pi value = %f\n",pi);

    //Method 1
    sum = 0.0;
    #pragma omp parallel num_threads(4) reduction(+:sum) shared(num_steps) private(i) default(none)
    {
        #pragma omp for
        for(i=0;i<num_steps;i++)
        {
            sum +=  4.0/(1.0 +  (i/(double)num_steps) * (i/(double)num_steps));
        }
    }
    pi = step*sum;
    printf("pi value = %f\n",pi);

    //Method 2
    sum = 0.0;
    #pragma omp parallel for num_threads(4) reduction(+:sum) shared(num_steps) private(i) default(none)
    for(i=0;i<num_steps;i++)
    {
        sum +=  4.0/(1.0 +  (i/(double)num_steps) * (i/(double)num_steps));
    }
    
    pi = step*sum;
    printf("pi value = %f\n",pi);


}
