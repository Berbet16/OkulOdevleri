# -*- coding: utf-8 -*-
import time

n = int(input("how many numbers to enter?"))

a = time.time()

array = []

for x in range (n):
    array.append(int(input()))
    
def shellSort(array, n):
    space = n  //2
    while space > 0:
        for i in range(space, n):
            temp = array[i]
            j = i
            while j >= space and array[j - space] > temp:
                array[j] = array[j - space]
                j -= space
            array[j] = temp
        space //=2

size = len(array)
shellSort(array, size)
print("Shell Sort:")
print(array)
print(time.time()-a)
#O(n2)