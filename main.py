import numpy as np
import math


def print_float_matrix(m):
    a, b = m.shape
    for i in range(a):
        print("[", end=' ')
        for j in range(b):
            print("%7.3f" % m[i][j], end=' ')
        print("]")
    print(" ")


def task_1():
    print("===================================\nTASK 1\n")

    m = n = 5
    A = np.random.randint(5, 11, (m, n))
    print("A =")
    print(A, "\n")

    U = np.zeros((m, n), float)
    L = np.identity(n, float)

    for i in range(m):
        for j in range(n):
            if i <= j:
                U[i, j] = A[i, j] - np.dot(L[i, :i], U[:i, j])
            if i > j:
                L[i, j] = (A[i, j] - np.dot(L[i, :j], U[:j, j])) / U[j, j]

    print("L =")
    print_float_matrix(L)

    print("U =")
    print_float_matrix(U)

    print("det A = det L * det U\n"
          "det L =", round(np.linalg.det(L)), "==> det A = det U\n"
          "det U is equal product of all elements of the diagonal\n")

    diag_prod = round(U.diagonal().prod())
    print("det U =", diag_prod)
    print("det A == det U :", round(np.linalg.det(A)) == diag_prod, "\n")
    return A


def task_2(A):
    print("===================================\nTASK 2\n")

    print("Gram–Schmidt process:\n")
    m, n = A.shape
    Q = np.zeros((m, n))
    R = np.zeros((n, n))

    for j in range(n):
        v = A[:, j]

        for i in range(j):
            q = Q[:, i]
            R[i, j] = q.dot(v)
            v = v - R[i, j] * q

        norm = np.linalg.norm(v)
        Q[:, j] = v / norm
        R[j, j] = norm

    print("Q =")
    print_float_matrix(Q)

    print("R =")
    print_float_matrix(R)

    QR = np.dot(Q, R)
    QR = QR.round()
    print("A == QR\n", A == QR)

    print("\nnp.linalg.qr(A):\n")
    Q, R = np.linalg.qr(A)

    print("_Q =")
    print_float_matrix(Q)

    print("_R =")
    print_float_matrix(R)

    _QR = np.dot(Q, R)
    _QR = _QR.round()
    print("_QR = _Q * _R\n_QR == QR\n", QR == _QR)


def accuracy_check(x_prev, x):
    sum_up = 0
    sum_low = 0
    for k in range(0, len(x_prev)):
        sum_up += (x[k] - x_prev[k]) ** 2
        sum_low += (x[k]) ** 2

    return math.sqrt(sum_up / sum_low) < 0.001


def task_3():
    print("===================================\nTASK 3\n")

    a = np.array([[3.1, 2.8, 1.9],
                  [1.9, 3.1, 2.1],
                  [7.5, 3.8, 4.8]], float)
    print("A =")
    print_float_matrix(a)

    b = np.array([[0.2], [2.1], [5.6]], float)
    print("B =")
    print_float_matrix(b)

    x = np.array([[0], [0], [0]], float)

    count = 0
    while (count < 1000):
        x_prev = x.copy()

        for k in range(0, 3):
            S = 0
            for j in range(0, 3):
                if (j != k):
                    S = S + a[k][j] * x[j]
            x[k] = b[k] / a[k][k] - S / a[k][k]

        if accuracy_check(x_prev, x):
            break

        count += 1

    print("Total iteration count =", count)
    x = x.copy()
    print("X =")
    print_float_matrix(x)

    print("np.linalg.solve(a, b) =")
    print_float_matrix(np.linalg.solve(a, b))


task_2(task_1())
task_3()
