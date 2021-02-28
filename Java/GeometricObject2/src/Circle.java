
public class Circle extends GeometricObject
{
	private double radius;
	//default constructor
	public Circle()
	{
	}
	public Circle(double radius)
	{
		this.radius = radius;
	}
	//getter and setter method
	public double getRadius() {
		return radius;
	}
	public void setRadius(double radius){
		this.radius = radius;
	}
	//find the area
	public double getArea()
	{
		return radius * radius * Math.PI;
	}
	//find the meter
	public double getPerimeter()
	{
		return 2* radius * Math.PI;
	}
	//print of what
	public String toPrint()
	{
		return "[Circle]";
	}
	//color and filled spelling
	@Override
	public String toString()
	{
		return  "color: " + color + " and filled: " + filled + "\nradius=" + radius;
	}
	
}
