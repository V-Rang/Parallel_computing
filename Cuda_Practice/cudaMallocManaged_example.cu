/*
Example of cudaMallocManaged  -> memory pointed to by both - cpu and device.

*/
#include<iostream>
#include<cuda.h>
#include<device_launch_parameters.h>

using namespace std;


__global__ void modder(int *a)
{
  printf("current value = %d\n",a[0]);
  a[0] = 17;
}

int main()
{
  
  int *y = new int[1]{14};
  cudaMallocManaged(&y,1*sizeof(int)); 
  
  /*y now pointed to by both cpu and device, if following line is commented and code run as is, a[0] in the above
  modder function is not 14.
  */

  y[0] = 24;
  modder<<<1,1>>>(y);
  cudaDeviceSynchronize();
  cout << y[0] << endl;

  return 0;
}
