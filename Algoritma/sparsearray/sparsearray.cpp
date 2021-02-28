#include<stdio.h>

#pragma warning(disable:4996)
int main()
{
	int i, j, n, temp;
	int numberArray[1000];
	int newArray[1000];
	printf("enter the number of numbers:\n");
	scanf("%d", &n);

	//reading the array
	for (int i = 0;i < n ;i++)
	{
		printf("enter the number:\n");
		scanf("%d", &numberArray[i]);
	}

	// sorting array
	for (int i = 0; i < (n - 1); i++) 
	{
		for (j = 0; j < (n - 1 - i); j++)
		{
			if (numberArray[j] > numberArray[j + 1])
			{
				temp = numberArray[j + 1];
				numberArray[j + 1] = numberArray[j];
				numberArray[j] = temp;
			}
		}
	}
	// printing sorted array
	printf("Array :\n");
	for (int i = 0;i < n;i++)
	{
		printf("> %d ", numberArray[i]);
	}
	printf("\n");
	//Sparse Array olusturma
	printf("Sparse Array number  %d \n  ", numberArray[n - 1]+1);

	for (int i = 0; i < (numberArray[n-1] +1);i++)
	{
		newArray[i] = 0;
		for (int j = 0; j < n; j++)
		{
			if (i == numberArray[j])
			{
				newArray[i] = 1;
			}
		}
	}
	// printing new array
	printf("New Array :\n");
	for (int i = 0;i < (numberArray[n-1]+1) ;i++)
	{
		printf("> %d      ", newArray[i]);
	}
	printf("\n");
}