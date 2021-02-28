# -*- coding: utf-8 -*-
import time

n = int(input("how many numbers to enter?"))

array = []
a = time.time()
for x in range (n):
    array.append(int(input()))

for i in range(len(array)):
    temp = array[i]
    j = i-1
    while j >=0 and temp < array[j] :
        array[j+1] = array[j] 
        j = j - 1
        array[j+1] = temp
print ("Sorted array is:") 
print(array)
print(time.time()-a)
#O(n2)