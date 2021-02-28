class Account 
{
	private int id = 0;
    private double balance = 0.0;
    private java.util.Date dateCreated;
    //Creates a default account
    public Account() 
    {
    	dateCreated = new java.util.Date();
    }
    //Creates a specified id and balance
    public Account(int id, double balance) 
    {
        this.id = id;
        this.balance = balance;
    }
    //getter methods id and balance
    public double getId()
    {
        return this.id;
    }

    public double getBalance() 
    {
        return this.balance;
    }
    //setter methods id and balance
    public void setId(int id) 
    {
        this.id = id;
    }

    public void setBalance(double balance) 
    {
        this.balance = balance;
    }
    //getter method of dateCreated
    public String getDateCreated() 
    {
        return this.dateCreated.toString();
    }
    //withdraw a specified amount
    public void withDraw(double amount) 
    {
        this.balance =this.balance - amount;
    }
    public void deposit(double amount) 
    {
        this.balance =this.balance + amount;
    } 
}
