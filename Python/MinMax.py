# -*- coding: utf-8 -*-

array = []
result = int(input("How many values to enter?" ))
for x in range(result) :
    arr = int(input("enter the number="))
    array.append(arr)
    
lowNumber = array[0]
highNumber = array[0]

i = result

for i in range(0,i):
    if lowNumber > array[i]:
        lowNumber = array[i]
    if highNumber < array[i]:
        highNumber = array[i]
        
print("High Number : {0} " .format(highNumber))
print("lowNumber : {0} " .format(lowNumber))

