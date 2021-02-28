# -*- coding: utf-8 -*-
import time
import matplotlib.pyplot as plt

n = int(input("How many numbers to enter?  "))  
  
array = []  
  
a=time.time()

for x in range(n):  
    array.append(int(input()))  
  
  
for i in range(n):  
     for j in range(n-1):  
        if array[i] < array[j]:  
            temp = array[j]  
            array[j] = array[i]  
            array[i] = temp  
  

print("Bubble Sort: ")  
print(array)
print(time.time()-a)
plt.show()