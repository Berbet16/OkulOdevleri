

def func( x ): 
	return 3* x * x * x + x * x -x -5

def derivFunc( x ): 
	return 3 * x * x - 2 * x -1

# Function to find the root 
def newtonRaphson( x ): 
	h = func(x) / derivFunc(x) 
	while abs(h) >= 0.0001: 
		h = func(x)/derivFunc(x)  
		x = x - h 
	print("The value of the root is : ", "%.4f"% x) 
x0 = -1 
newtonRaphson(x0) 
