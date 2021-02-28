import java.util.Scanner;
public class ATM
{
	private static Scanner scan = new Scanner(System.in);
	int id;
	Account[] accounts;
	public ATM()
	{
		//Creates ten Account in an array
		accounts = new Account[10];
		for(int i=0;i<accounts.length;i++)
		{
			accounts[i]=new Account(i, 100.0);
		}
		mainMenuOptions();
	}
	public void mainMenuOptions()
	{
		//Enter an id
		System.out.println("enter the id:");
		id=scan.nextInt();
		if(id>1 || id<10)
		{
			correctId(id);
		}
		while(true)
		{
			menuDisplay();
			System.out.println("enter the choice:");
			int choice=scan.nextInt();
			if (choice == 4)
			{
                if (id < 1 || id > 10)
                {
                    id = correctId(id);
                }
                mainMenuOptions();
			}
			performChoice(id,choice,accounts);
		}
	 }
	//Enter a choice by the user
	private static void performChoice(int id,int choice , Account[] accounts)
	{
		switch(choice)
		{
            case 1:
			    System.out.println("The balance is " + accounts[id].getBalance());
                break;
            case 2:
            	System.out.print("Enter an amount to withdraw:");
    	    	double withdraw= scan.nextInt();
    	    	accounts[id].setBalance(accounts[id].getBalance()-withdraw);
                break;
	        case 3:
            	System.out.print("Enter an amount to deposited:");
                double deposit = scan.nextInt();	
                accounts[id].setBalance(accounts[id].getBalance()+deposit);
                break;
            case 4:
            	System.exit(4);
        }
	}
	//if id is not correct
	public static int correctId(int id) 
	{
        while (id < 1 || id > 10)
        {
            System.out.print("Please enter the correct id(1-10) : ");
			id = scan.nextInt();
        }
        return id;
        }  
	//Menu prompts the user to enters
	public static void menuDisplay() 
	{
		System.out.println(" -Main menu-");
        System.out.println("1: check balance");
        System.out.println("2: withdraw");
        System.out.println("3: deposit");
        System.out.println("4: exit");	
	}
}