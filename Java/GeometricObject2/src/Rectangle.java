
public class Rectangle extends GeometricObject
{
	private double width;
	private double height;
	//default access
	public Rectangle() 
	{
	}
	public Rectangle(double width, double height)
	{
		this.width = width;
		this.height = height;
	}
	//getters and setters method
	public double getWidth() {
		return width;
	}
	public void setWidth(double width){
		this.width = width;
	}
	public double getHeight() {
		return height;
	}
	public void setHeight(double height) {
		this.height = height;
	}
	//find the area
	public double getArea() 
	{
	return width * height;
	}
	//find the meter
	public double getPerimeter()
	{
		return 2 * (width + height);
	}
	//print of what
	public String toPrint()
	{
		return "[Rectangle]";
	}
	//color and filled spelling
	@Override
	public String toString()
	{
		return   "color: " + color + " and filled: " + filled + "\nwidth=" + width +"height=" + height;
	} 
}
