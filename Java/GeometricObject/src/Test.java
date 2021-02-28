import java.util.Scanner;
public class Test 
{
	public static void main(String[] args)
	{
		Scanner scan =new Scanner(System.in);
    	System.out.println("Enter three sides:");
    	double side1 = scan.nextDouble();
    	double side2 = scan.nextDouble();
    	double side3 = scan.nextDouble();
    	
    	System.out.println("Enter the color:");
		String color = scan.next();
    	
		System.out.println("Enter a boolean value for filled:");
		String filled = scan.next();
		
    	Triangle triangle = new Triangle(side1 , side2, side3);
    	
    	System.out.println("TRÝANGLE CLASS:" + " Triangle: side 1: " + side1 + " Side 2: " + side2 + " Side 3: " + side3);
    	System.out.println("The area is " + triangle.getArea() );
    	System.out.println("The perimeter is " + triangle.getPerimeter());	
    	
    	java.util.Date date = new java.util.Date();
    	GeometricObject object =new GeometricObject();
    	
    	//makes toString calls in the triangle class
    	System.out.println("GEOMETRIC OBJECT CLASS:" + "Created on " + date.toString());
		System.out.println(triangle.getColor());
		System.out.println(triangle.getFilled());
		
		System.out.println("----------OUTPUT OF POLIMORPHISM EXAMPLE--------------");
		System.out.println("TRÝANGLE CLASS:" + triangle.toString());
		System.out.println("The area is " + triangle.getArea());
		System.out.println("The perimeter is " + triangle.getPerimeter());
		System.out.println("GEOMETRIC OBJECT CLASS:" + object.toString());
    }
}
