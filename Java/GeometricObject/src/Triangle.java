public class Triangle extends GeometricObject 
{
    public double side1=1.0;
    public double side2=1.0;
    public double side3=1.0;
	//default constructor
    public Triangle() 	
    {
		double side1=1.0;
		double side2=1.0;
		double side3=1.0;
    }
    //parametric constructor
    public Triangle(double side1,double side2,double side3)
    {
    	this.side1=side1;
    	this.side2=side2;
    	this.side3=side3;
    }
    //The access methods
    public double getSide1()
    {
        return side1;
    }
    public double getSide2() 
    {
        return side2;
    }
    public double getSide3() 
    {
        return side3;
    }
    public void setSide1(double side1)
    {
        this.side1 = side1;
    }

    public void setSide2(double side2)
    {
        this.side2 = side2;
    }

    public void setSide3(double side3)
    {
        this.side3 = side2;
    }
    // function for getting area of triangle
    public double getArea() 
    {
    	double s = getPerimeter();
    	return Math.sqrt(s * (s - side1) * s*(s - side2) * s*(s - side3));
    }
    //function for getting parametric for triangle
    public double getPerimeter()
    {
    	double s = (side1 + side2 + side3)/2;
        return s;
    }
    // print the data
    public String toString()
    {
        return " Triangle: Side 1 = " + side1 + " Side 2 = " + side2 + " Side 3 = " + side3;
    }
}









