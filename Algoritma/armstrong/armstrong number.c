#include<stdio.h>
#pragma warning(disable:4996)
int main()
{
	int a,b,i,temp,tempnumber,tempnumber2,power=0,sum=0;
	printf("a:");
	scanf("%d",&a);
	printf("b:");
	scanf("%d",&b);
	for(i=a;i<=b;i++)
	{
		power=0;
		sum=0;
		tempnumber=i;
		while(tempnumber>=1)
		{
			tempnumber=tempnumber/10;
			power=power+1;
		}
		while(tempnumber2>=1)
		{
			temp=tempnumber%10;
			sum=sum+pow(temp,power);
			tempnumber2=tempnumber2/10;
			
		}
		if(sum==i)
		{
			printf("%d is an armstrong number.\n",i);
		}
		else
		{
			printf("%d is not an armstrong number",i);
		}
	}
	return 0;
}
