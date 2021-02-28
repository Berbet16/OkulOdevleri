
import java.util.Scanner;
public class Arrays
{
	public static void main(String[] args)
	{
		//Determine matrix1 dimensions
		System.out.println("Enter the number of rows of first matrix:");
		Scanner size =new Scanner(System.in);
		int x = size.nextInt();
		System.out.println("Enter the number of columns of first matrix:");
		int y = size.nextInt();
		
		//Printing the values of matrix1
		int matrix1[][] = new int[x][y];
		Scanner elements = new Scanner(System.in);
		
		for(int i=0 ; i<x ;i++)
		{
			for(int j=0 ;j<y; j++)
			{
				System.out.println("matrix1:" + "[" + (i+1) + "]" + "[" + (j+1) + "]" + "=");
				matrix1[i][j] = elements.nextInt();
			}
		}
		System.out.println("Enter the elements of first matrix:" );
		for(int i=0 ; i<x ;i++)
		{
			for(int j=0 ;j<y; j++)
			{
				System.out.print(matrix1[i][j]  +  "    ");
			}
			System.out.println();
		}
		
		//Determine matrix2 dimensions
		System.out.println(" Enter the number of rows of second matrix:");
		int t = size.nextInt();
		System.out.println(" Enter the number of columns of first matrix:");
		int k = size.nextInt();
		
		//Printing the values of matrix2
		int matrix2[][] = new int[y][t];
		for(int i=0 ; i<y ;i++)
		{
			for(int j=0 ;j<t; j++)
			{
				System.out.println("matrix2:" + "[" + (i+1)  + "]" +"[" + (j+1) + "]" + "=");
				matrix2[i][j] = size.nextInt();
			}
		}
		System.out.println("Enter the elements of second matrix:" );
		for(int i=0 ; i<y ;i++)
		{
			for(int j=0 ;j<t; j++)
			{
				System.out.print(matrix2[i][j] + "    ");
			}
			System.out.println();
		}
		//Printing of result
		if(y==t) 
		{
		    int matrix3[][] = new int[x][k];
		    for(int i=0; i<x ; i++)
		    {
			
			    for(int j=0; j<k ;j++) 
			    {
				    for(int k1=0; k1<k ;k1++)
				    {
					   matrix3[i][j] = matrix3[i][j] + (matrix1[i][k1] * matrix2[k1][j]);
				    }
			    }
		     }
		    System.out.println();
			System.out.println("Result:");
			for(int i = 0 ; i<x ; i++)
			{
				for(int j=0 ; j<k ; j++)
				{
					System.out.print(matrix3[i][j] + "    ");
				}
				System.out.println();
			}
		}
		
		else
		{
	        System.out.println("We can't do this multiplication!");
		}
	}
}