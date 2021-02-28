# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 17:43:51 2019

@author: BS
"""



def factorial(n):
    print("factorial has been called with n = " + str(n))
    if n == 1:
        return 1
    else:
        res = n * factorial(n-1)
        print("intermediate result for ", n, " * factorial(" ,n-1, "): ",res)
        return res	

print(factorial(5))
