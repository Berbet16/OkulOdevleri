
public abstract class GeometricObject 
{
	protected String color = "white";
	protected boolean filled;
	private final java.util.Date dateCreated;
	//constructors
	protected GeometricObject()
	{
		dateCreated = new java.util.Date();
	}
	protected GeometricObject(String color, boolean filled)
	{
		dateCreated = new java.util.Date();
		this.color = color;
		this.filled = filled;
	}
	//getters and setters methods
	public String getColor() {
		return color;
	}
	public void setColor(String color) {
		this.color = color;
	}
	public boolean getFilled() {
		return filled;
	}
	public void setFilled(boolean filled) {
		this.filled = filled;
	}
	public java.util.Date getDateCreated() {
		return dateCreated;
	}
	public abstract double getArea();
	public abstract String toPrint();
}
