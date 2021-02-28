# -*- coding: utf-8 -*-
import time

n = int(input("how many numbers to enter?"))
array = []
a=time.time()
for x in range (n):
    array.append(int(input()))

for i in range(len(array)):
    for j in range(i+1 ,len(array)):
        if array[j] < array[i]:
            min = array[j]
            array[j] = array[i]
            array[i] = min
            
print("Selection Sort:")

for z in range(n):
    print(array[z])   

print(array)
print(time.time()-a)
#O(n2)