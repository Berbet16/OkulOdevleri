

# Function to find the product term  
def proterm(i, value, x):  
    pro = 1;  
    for j in range(i):  
        pro = pro * (value - x[j]);  
    return pro;  

# divided difference table  
def dividedDiffTable(x, y, n): 
  
    for i in range(1, n):  
        for j in range(n - i):  
            y[j][i] = ((y[j][i - 1] - y[j + 1][i - 1]) / (x[j] - x[i + j])); 
    return y; 

# divided difference formula  
def applyFormula(value, x, y, n):  
  
    sum = y[0][0];  
  
    for i in range(1, n): 
        sum = sum + (proterm(i, value, x) * y[0][i]);  
      
    return sum;  
 
# difference table  
def printDiffTable(y, n):  
  
    for i in range(n):  
        for j in range(n - i):  
            print(round(y[i][j], 4), "\t",  
                               end = " ");  
  
        print("");  

# number of inputs given  
n = 4;  
y = [[0 for i in range(10)]  
        for j in range(10)];  
x = [ -2,0 ,1 , 3];  
  
y[0][0] = 8;  
y[1][0] = 4;  
y[2][0] = 2;  
y[3][0] = -2;  

y=dividedDiffTable(x, y, n);  

printDiffTable(y, n);  

value = 0;  

print("\nValue at", value, "is", round(applyFormula(value, x, y, n), 2)) 
