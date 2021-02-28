#include <stdio.h>
#pragma warning(disable:4996)
int main()
{
 int Array[100][100], A, B, RowsCounter, ColumnsCounter;
 printf("Please the number of rows \n");
 scanf("%d", &A);
 printf("Please the number of columns \n");
 scanf("%d", &B);
 for (RowsCounter = 0; RowsCounter < A; RowsCounter++)
  for (ColumnsCounter = 0; ColumnsCounter < B; ColumnsCounter++)
	{
	 printf("Please enter an elements for row= %d and column= %d: ", RowsCounter, ColumnsCounter);
	 scanf("%d", &Array[RowsCounter][ColumnsCounter]);
	} 
	// Printing the array elements

 system("pause");
 return 0;
}
