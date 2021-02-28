#181805028_BETÜL_BAŞAN
#171805019_BETÜL_BERNA_SOYLU

def func(x):
    return x**2

lower_limit = 0  # Lower limit
upper_limit = 1  # Upper limit

# Function for approximate integral
def simpsons_(ll, ul, n):
    # Calculating the value of h
    h = (ul - ll) / n
    # List for storing value of x and f(x)
    x = list()
    fx = list()

    # Calcuting values of x and f(x)
    i = 0
    while i <= n:
        x.append(ll + i * h)
        fx.append(func(x[i]))
        i += 1

    # Calculating result
    res = 0
    i = 0
    while i <= n:
        if i == 0 or i == n:
            res += fx[i]
        elif i % 2 != 0:
            res += 4 * fx[i]
        else:
            res += 2 * fx[i]
        i += 1
    res = res * (h / 3)
    return res

print("two panel ")
print(simpsons_(lower_limit, upper_limit, 2))
print("four panel ")
print(simpsons_(lower_limit, upper_limit, 4))
print("six panel ")
print(simpsons_(lower_limit, upper_limit, 6))