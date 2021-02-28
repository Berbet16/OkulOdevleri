
public class BMItest
{
		public static void main(String[] args)
		{
			BMI bmiWithoutArgument = new BMI( );
			System.out.print("The BMI for ");
			System.out.print(bmiWithoutArgument.getName());
			System.out.print("	is ");
			System.out.print(bmiWithoutArgument.getBMI());
			System.out.print("	");
			System.out.println(bmiWithoutArgument.getStatus());
			
			
			BMI bmiDefaultAge = new BMI( "Sarah King", 215 ,70 );
			System.out.print("The BMI for ");
			System.out.print(bmiDefaultAge.getName());
			System.out.print("	is ");
			System.out.print(bmiDefaultAge.getBMI());
			System.out.print("	");
			System.out.println(bmiDefaultAge.getStatus());
			
			BMI bmiSpecifiedArguments = new BMI("Kim Young", 18, 145, 70);
			System.out.print("The BMI for ");
			System.out.print(bmiSpecifiedArguments.getName());
			System.out.print("	is ");
			System.out.print(bmiSpecifiedArguments.getBMI());
			System.out.print("	");
			System.out.println(bmiSpecifiedArguments.getStatus());
		}
}