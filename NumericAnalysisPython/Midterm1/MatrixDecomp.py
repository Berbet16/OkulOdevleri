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


def doolittle(A):
    N = len(A)
    L = np.zeros([N, N])
    U = np.zeros([N, N])
    for i in range(N):
        for k in range(i, N):
            for j in range(i, N):
                Utotal = 0
                for j in range(i):
                    Utotal += L[i, j] * U[j, k]
                U[i, k] = A[i, k] - Utotal
                if (i == k):
                    L[i, k] = 1
                else:
                    Ltotal = 0
                    for j in range(i):
                        Ltotal += L[k, j] * U[j, i]
                    L[k, i] = (A[k, i] - Ltotal) / U[i][i]
    return (L, U)


def computing_final_solution(A, b, algorithm_used):
    L, U = algorithm_used(A)

    print("L = " + str(L) + "\n")
    print("U = " + str(U) + "\n")

    y = forward_substitution(L, b)
    x = backward_substitution(U, y)

    return x


A = np.array([[2.34, -4.10, 1.78], [-1.98, 3.47, -2.2], [2.36, -15.17, -6.18]])
b = np.array([0.02, -0.73, -6.63])

print("The solution using Doolittle's algorithm:" + "\n")
print("x = " + str(computing_final_solution(A, b, doolittle)) + "\n")
