
public class BMI
{
	private String name;
	private int age;
	private double weight;  //in pound
	private double height;  //in inch
	public static final double KILOGRAMS_PER_INCH=0.45359237;
	public static final double METERS_PER_INCH=0.0254;
	// Create a class constructor for the BMI class without parameters  
	public BMI() 
	{
	    name = "John Black";
            age = 20;
	    weight = 100;
	    height = 50;
	} 
	// Create a class constructor for the BMI with default age 20 other information is specified 
 	public BMI(String specifiedName, double specifiedWeight, double specifiedHeight)
 	{
	    name = specifiedName; 
 	    age = 20;
 	    weight = specifiedWeight;
 	    height = specifiedHeight;
        }
	// Create a class constructor for the BMI with specified information
	public BMI(String specifiedName, int specifiedAge, double specifiedWeight, double specifiedHeight)
	{
	    name = specifiedName; 
            age = specifiedAge;
            weight = specifiedWeight;
            height = specifiedHeight;
        }
	// getter Methods
	public String getName()
	{
	    return name;
	}
	public int getAge()
	{
	    return age;
	}
	public double getWeight()
	{
	    return weight;
	}
	public double getHeight()
	{
	    return height;
	}
	// setter Methods
	public String setName(String specifiedName)
	{
	    return name = specifiedName;
	}
	public int setAge(int specifiedAge)
	{
	    return age = specifiedAge;
	}
	public double setWeight(double specifiedWeight)
	{
	    return weight = specifiedWeight;
	}
	public double setHeight(double specifiedHeight)
	{
	    return height = specifiedHeight;
	}

    // BMI Calculator getBMI 
    public double getBMI()
    {
     	double bmi = (weight*KILOGRAMS_PER_INCH) / ((height*METERS_PER_INCH) * (height*METERS_PER_INCH));
    	return Math.round(bmi * 100) / 100.0;
    }

    //get BMIStatus //
 	public String getStatus()
	{
		double bmi = getBMI();
		if(bmi<=18)
		{
			return "underweight";
		}
		else if(bmi>18 && bmi<=24)
		{
			return "normal weight";
		}
		else if(bmi>24 && bmi<=29)
		{
			return "overweight";
		}
		else
		{
			return "obese";
		}
	}
}
