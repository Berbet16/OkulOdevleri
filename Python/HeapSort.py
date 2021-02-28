# -*- coding: utf-8 -*-
n = int(input("how many numbers to enter?"))
array=[]
for x in range (n):
    array.append(int(input()))
def heapify(array, n, i): 
    largest = i   
    l = 2 * i + 1     
    r = 2 * i + 2    
    if l < n and array[i] < array[l]: 
        largest = l 
    if r < n and array[largest] < array[r]: 
        largest = r 
    if largest != i: 
        array[i],array[largest] = array[largest],array[i]  
        heapify(array, n, largest) 
def heapSort(array): 
    n = len(array)  
    for i in range(n, -1, -1): 
        heapify(array, n, i)  
    for i in range(n-1, 0, -1): 
        array[i], array[0] = array[0], array[i]   
        heapify(array, i, 0) 
heapSort(array)  
print ("Sorted array is",array)  