
public class Main 
{
	public static void main(String[] args) 
	{
		// Create an array of five GeometricObjects
		GeometricObject[] objects = new GeometricObject[5];
		objects[0] = new Square(2.0);
        objects[1] = new Circle(5.0);
        objects[2] = new Square(5.0);
        objects[3] = new Rectangle(3,4);
        objects[4] = new Square(4.5);
        
        //Display the result
        for (int i = 0; i < objects.length; i++) 
        {
        	System.out.println(objects[i].toPrint() + "Created on " + objects[i].getDateCreated());
        	System.out.println(objects[i].toString());
        	System.out.println("Area of objects is " + objects[i].getArea());
		if (objects[i] instanceof Colorable)
        {
			((Colorable) objects[i]).howToColor();
        }
        }
	}

}
