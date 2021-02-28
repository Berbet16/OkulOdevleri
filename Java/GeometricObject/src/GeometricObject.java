
public class GeometricObject 
{
	public String color = "white " ;
	public boolean filled;
	public java.util.Date dateCreated;
	//A no-arguments constructor
	public GeometricObject()
	{
		super();
        dateCreated = new java.util.Date();
    }
	//A constructor with parameters
	public GeometricObject(String color, boolean filled)
	{
		dateCreated = new java.util.Date();
		this.color=color;
		this.filled=filled;
	}
	//The getter methods for all three data fields
	public String getColor()
	{
        return color;
    }
	public boolean getFilled() 
	{
		boolean filled = true;
        return filled;
    }
	public java.util.Date getDateCreated() 
	{
        return dateCreated;
    }
	//the setter methods for color and filled data fields
	public void setColor(String color) 
	{
        this.color = color;
    }
	public void setFilled(boolean filled) 
	{
        this.filled = filled;
    }
	@Override
	public String toString() 
	{
		return "Created on : " + dateCreated   + " \ncolor: "  + color + "and" +  "  filled=" + filled;
	}
	
	
	
}
