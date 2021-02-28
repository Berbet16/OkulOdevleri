
public class Square extends GeometricObject implements Colorable
{
	public double side;
	//default constructor
	public Square() 
	{
    }
	public Square(double side)
	{
		this.side = side;
	}
	//getter and setter method
	public double getSide() {
		return side;
	}
	public void setSide(double side) {
		this.side = side;
	}
	//access to constructor
	@Override
	public void howToColor()
	{
	    System.out.println("Color all four sides");
	}
	//find the area
	@Override
	public double getArea() 
	{
		double area = side * side;
		return area;
	}
	public double getPerimeter()
	{
		return 4 * side;
	}
	//print of what
	public String toPrint()
	{
		return "[Square]";
	}
	//color and filled spelling
	@Override
	public String toString()
	{
		return  "color: " + color + " and filled: " + filled + " \nside:" + side ;
	}
}