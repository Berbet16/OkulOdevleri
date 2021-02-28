#include<stdio.h>
#pragma warning(disable:4996)
int main()
{
	int n,i,sum=0,prime=0;
	printf("enter the n:");
	scanf("%d",&n);
	for(i=2;i<n;i++)
	{
		if(n%i==0)
		{
			i++;
			break;
		}
	    if(prime==0)
		{
			printf("n is prime number",n);
		}
		else
		{
			printf("n is not a prime number",n);
		}
		
	}
	return 0;
	
}

