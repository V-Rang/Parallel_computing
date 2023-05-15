/*
Atomic directive: 
Template:

#pragma omp atomic
x <binop> = expr;
#pragma omp atomic

// one of
x++, ++x, x-- or --x

binop: 4 arithmetic: +, -, *, /
       5 bitwise: |, &, ^, <<, >>
|: bitwise or 
&: bitwise and
^: bitwise XOR-> 0 if X,Y are the same, else 1
<<: bitwise left shift -> move to left, pad with zeros 4 << 1 = 8
>>: bitwise right shift -> move to right, pad with zeros 3 >> 1 = 1
Taken from: https://learn.microsoft.com/en-us/cpp/cpp/left-shift-and-right-shift-operators-input-and-output?view=msvc-170
*/


#include<iostream>
#include<omp.h>
#include<climits>
#include<bitset>

using namespace std;


int main()
{
    omp_set_num_threads(5);
    int tid;
    int i,j;

    unsigned short short1 = 3;
    unsigned short short2 = short1>>1;
    cout << short2 << endl;


    return 0;
}