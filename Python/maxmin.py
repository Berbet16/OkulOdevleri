# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 17:37:23 2019

@author: BS
"""

x=int(input("Kac tane deger gireceksiniz?"))
sayilar=[]
for i in range(x):
    sayi = int(input("Sayiyi giriniz:"))
    sayilar.append(sayi)
print ("En buyuk sayi:", max(sayilar),"\t","En kucuk sayi:",min(sayilar))