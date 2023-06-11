/*
scanf - needs location/reference to which a value needs to be assigned to.
*/

#include<iostream>

using namespace std;

int main()
{
    int a,*b;

    printf("Enter the value of a\n");
    fflush(stdout);
    scanf("%d",&a);
    printf("Value of a = %d\n",a);


    b = new int[1]; //code will not work if you don't do this. b has to "point" to a location in memory to intialize a value in that location.
    printf("Enter the value of b\n");
    fflush(stdout);
    scanf("%d",b);
    printf("Value of b = %d\n",*b);

    return 0;
}