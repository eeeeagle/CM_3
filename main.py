import numpy as np


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

    m = n = 8
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


def task_3():
    print("===================================\nTASK 3\n")

    a = np.array([[3.1, 2.8, 1.9],
                  [1.9, 3.1, 2.1],
                  [7.5, 3.8, 4.8]], float)

    b = np.array([[0.2], [2.1], [5.6]], float)


task_2(task_1())
task_3()
