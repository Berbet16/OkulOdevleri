


sayi =int(input("sayi:"))

factorial = 1

if sayi<0:
    print("negatif sayilarda faktoriyel olmaz.")
elif sayi ==0:
    print("sonuc:1")
else:
    for i in range (1,sayi+1):
        factorial=factorial*i
    print("sonuc:",factorial)
    