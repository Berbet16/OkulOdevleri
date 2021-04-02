#continue kullanmamızın amacı 3 e geldiğinde fizz yazsa da devamında 3 gösteriyor fakat eğer continue
#kullanırsak 3 ü atlayarak sıradaki sayıya devam ediyor.

# 1-100 arasındaki sayılar için
for number in range(1, 100):
    #3'e bölünen sayılar için
    if number % 3 ==0:
        print("Fizz")
        continue
    elif number % 5 == 0:
        print("Buzz")
        continue
    elif number % 15 == 0:
        print("FizzBuzz")
        continue
    print(number)