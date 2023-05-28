/*
Difference between stack and heap storage:
https://stackoverflow.com/questions/216259/is-there-a-max-array-length-limit-in-c

*/

#include<iostream>

using namespace std;

int main()
{
   
    // int b[(int)1e6];             //if uncommented, the hello world is never printed.
    int *a = new int[(int)1e6];     // if uncommented, hello world below is printed.


    printf("Hello world\n");
    return 0;
}