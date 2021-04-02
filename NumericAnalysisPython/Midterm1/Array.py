import numpy as np

A = np.array([[2, -3, -1], [3, 2, -5], [2, 4, -1]])
print('A:', A)

b = np.array([3, -9, -5])
print('b:', b)

n = len(b)
x = np.zeros(n)

for k in range(n - 1):
    for i in range(k + 1, n):
        if A[i, k] == 0:continue
        factor = A[k, k] / A[i, k]
        for j in range(k, n):
            A[i, j] = A[k, j] - A[i, j] * factor
        b[i] = b[k] - b[i] * factor


x[n - 1] = b[n - 1] / A[n - 1, n - 1]
for i in range(n - 2, -1, -1):
    sum_Ax = 0
    for j in range(i + 1, n):
        sum_Ax += A[i, j] * x[j]
    x[i] = (b[i] - sum_Ax) / A[i, i]

print('***SOLUTION***')
print('x:', x)






