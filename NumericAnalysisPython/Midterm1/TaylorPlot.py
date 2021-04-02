import math

# Taylor expansion at n=1 2x
# Taylor expansion at n=3 2x - 4(x**3)/3
# Taylor expansion at n=5 2x - 4(x**3)/3 + 4(x**5)/15

import sympy as sy
import numpy as np
from sympy.functions import sin, cos
import matplotlib.pyplot as plt
from sympy.parsing.sympy_parser import *

plt.style.use("ggplot")

# Define the variable and the function to approximate
x = sy.Symbol('x')
f = sin(2 * x)


# Factorial function
def factorial(n):
    if n <= 0:
        return 1
    else:
        return n * factorial(n - 1)


# Taylor approximation at x0 of the function 'function'
def taylor(function, x0, n):
    i = 0
    p = 0
    while i <= n:
        p = p + (function.diff(x, i).subs(x, x0)) / (factorial(i)) * (x - x0) ** i
        i += 1
    return p


# Plot results
def plot():
    x_lims = [0, math.pi]
    x1 = np.linspace(x_lims[0], x_lims[1], 800)
    y1 = []
    # Approximate up until 5 starting from 1 and using steps of 2
    lastFunc = 0
    for j in range(1, 6, 2):
        func = taylor(f, 0, j)
        lastFunc = func
        print('Taylor expansion at n=' + str(j), func)
        for k in x1:
            y1.append(func.subs(x, k))
        plt.plot(x1, y1, label='order ' + str(j))
        y1 = []
    # Plot the function to approximate (sine, in this case)
    expr = parse_expr(str(lastFunc))
    realValue = np.sin(math.pi / 2)  # sin(2x) => x=pi/2
    taylorValue = expr.subs(x, math.pi / 4)

    print('\n\nTaylor expansion sin of 2x=', lastFunc)
    print('Taylor expansion result sin of 2x ~=', taylorValue)
    print('Real value sin of 2x ~=', realValue)

    absoluteError = abs(realValue - taylorValue)
    realtiveError = absoluteError / realValue
    percentageError = realtiveError * 100

    print('\n\nAbsolute Error: ', absoluteError)
    print('Realtive Error: ', realtiveError)
    print('Percentage Error: "%"', percentageError)

    plt.plot(x1, np.sin(x1), label='sin of 2x')
    plt.xlim(x_lims)
    plt.ylim([-math.pi, math.pi])
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True)
    plt.title('Taylor series approximation')
    plt.show()


plot()
