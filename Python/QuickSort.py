# -*- coding: utf-8 -*-
n = int(input("how many numbers to enter?"))
array = []
for x in range (n):
    array.append(int(input()))
    
def partition(arr,low,high): 
    i = ( low-1 )
    pivot = arr[high]     
    for j in range(low , high):   
        if   arr[j] <= pivot: 
            i = i+1 
            arr[i],arr[j] = arr[j],arr[i] 
  
    arr[i+1],arr[high] = arr[high],arr[i+1] 
    return ( i+1 )  
def quickSort(arr,low,high): 
    if low < high: 
        pi = partition(arr,low,high)  
        quickSort(arr, low, pi-1) 
        quickSort(arr, pi+1, high) 

quickSort(array,0,n-1) 
print ("Sorted array is:",array) 
