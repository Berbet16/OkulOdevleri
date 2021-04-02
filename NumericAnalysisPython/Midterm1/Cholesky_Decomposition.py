import numpy as np

def forward_substitution(L, b):
    y = np.full_like(b, 0)
    for k in range(len(b)):
        y[k] = b[k]
        for i in range(k):
            y[k] = y[k] - (L[k, i] * y[i])
        y[k] = y[k] / L[k, k]
    return y

def backward_substitution(U, y):
    x = np.full_like(y, 0)
    for k in range(len(x), 0, -1):
        x[k - 1] = (y[k - 1] - np.dot(U[k - 1, k:], x[k:])) / U[k - 1, k - 1]
    return x

def cholesky_decomposition(A):
    n = len(A)
    L = np.ones((n, n)) * 0.0
    for i in range(n):
        for j in range(i + 1):
            if i == j:
                sum = 0
                for k in range(i):
                    sum = sum + L[i][k] ** 2
                L[i][i] = (A[i][i] - sum) ** 0.5
            else:
                sum = 0
                for k in range(j):
                    sum = sum + L[i][k] * L[j][k]
                L[i][j] = (A[i][j] - sum) / L[j][j]
    U = L.transpose()

    return L, U

def solution(A, b):

    y = forward_substitution(L, b)
    x = backward_substitution(U, y)

    return x


A = np.array([[1, 1, 1], [1, 2, 2], [1, 2, 3]])
print('A:', A)
b = np.array([1, 1.5, 3])
print('b:', b)
L, U = cholesky_decomposition(A)
print('L:', L)
print('U:', U)

print('***SOLUTION***')
x = solution(A, b)
print('x:', x)




