
# -*- coding: utf-8 -*-

#def sum(x,y):
#    return x+y
#def difference(x,y):
#    return x-y
#def multiplication(x,y):
#    return x*y
#def division(x,y):
#    return x/y
print("operations:")
print("1.addition")
print("2.subtraction")
print("3.multiplication")
print("4.division")
#input 
operation=input("select action:")

num1 = float(input("enter the first number:"))
num2 = float(input("enter the second number:"))

if operation == '1' :
    print("addition={:.5f}".format(num1+num2))
    
elif operation == '2' :
    print("subtraction={:.5f}".format(num1-num2))
    
elif operation == '3' :
    print("multiplication={:.5f}".format(num1*num2))
elif operation == '4' :
    print("division= {:.5f}".format(num1/num2))
else:
    print("no account")