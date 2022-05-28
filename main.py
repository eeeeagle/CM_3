import numpy as np


def print_float_matrix(m):
    a, b = m.shape
    for i in range(a):
        print("[", end=' ')
        for j in range(b):
            print("%8.4f" % m[i][j], end=' ')
        print("]")
    print(" ")


def task_1():
    print("===================================\nTASK 1\n")

    m = n = 7
    A = np.random.randint(2, 9, (m, n))
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

    print("det L =", round(np.linalg.det(L)))

    diag_prod = round(U.diagonal().prod())
    print("det U =", diag_prod)
    print("det A:", round(np.linalg.det(A)) == diag_prod, "\n")
    return A


def task_2(A):
    print("===================================\nTASK 2\n")
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
    print("QR =")
    print_float_matrix(QR)

    _Q, _R = np.linalg.qr(A)

    print("np.linalg.qr(A):\n_Q =")
    print_float_matrix(Q)

    print("_R =")
    print_float_matrix(R)

    _QR = np.dot(Q, R)
    _QR = _QR.round()
    print("_Q * _R = _QR\n_QR =")
    print_float_matrix(_QR)

    print("_QR = QR\n", _QR == QR, "\n")


def task_3():
    print("===================================\nTASK 3\n")

    a = np.array([[3.6, 1.8, -4.7], [2.7, -3.6, 1.9], [1.5, 4.5, 3.3]], float)
    print("A =")
    print_float_matrix(a)

    b = np.array([[3.8], [0.4], [-1.6]], float)
    print("B =")
    print_float_matrix(b)

    a[0] = a[0] + a[1]
    b[0] = b[0] + b[1]

    c = np.zeros((3, 3), float)
    f = np.zeros(3, float)

    for i in range(0, 3):
        for j in range(0, 3):
            if i == j:
                c[i, j] = 0
            else:
                c[i, j] = -a[i, j] / a[i, i]

    print("C =")
    print_float_matrix(c)

    for i in range(0, 3):
        f[i] = b[i] / a[i, i]

    F = np.array([[f[0]], [f[1]], [f[2]]])
    print("F =")
    print_float_matrix(F)

    x = np.zeros(3, float)
    x1 = f

    eps = 0.001
    while (abs(x1[0] - x[0]) > eps) | (abs(x1[1] - x[1]) > eps) | (abs(x1[2] - x[2]) > eps):
        tmp = np.dot(c, x1) + f
        x = x1
        x1 = tmp

    X = np.array([[x1[0]], [x1[1]], [x1[2]]])
    print("X =\n", X, "\n")

    print("np.linalg.solve(A, B)) =\n", np.linalg.solve(a, b))


task_2(task_1())
task_3()
