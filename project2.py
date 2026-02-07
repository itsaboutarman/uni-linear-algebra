import numpy as np
import numpy.linalg as LA
import time

N = 1000
M = np.zeros((N, N))
for k in range(N):
    M[k, k] = 5
    if k > 0:
        M[k, k-1] = -1
    if k < N-1:
        M[k, k+1] = -1

g = np.ones(N)

def chol(A):
    m = len(A)
    L = np.zeros_like(A)
    for r in range(m):
        for c in range(r+1):
            t = sum(L[r, z] * L[c, z] for z in range(c))
            if r == c:
                L[r, c] = np.sqrt(A[r, r] - t)
            else:
                L[r, c] = (A[r, c] - t) / L[c, c]
    return L

def f_sub(L, b):
    m = len(b)
    u = np.zeros(m)
    for r in range(m):
        u[r] = (b[r] - sum(L[r, z] * u[z] for z in range(r))) / L[r, r]
    return u

def b_sub(U, y):
    m = len(y)
    v = np.zeros(m)
    for r in range(m-1, -1, -1):
        v[r] = (y[r] - sum(U[r, z] * v[z] for z in range(r+1, m))) / U[r, r]
    return v

def thomas(A, f):
    n = len(f)
    a = np.zeros(n)
    b = np.zeros(n)
    c = np.zeros(n)
    for i in range(n):
        b[i] = A[i, i]
        if i > 0:
            a[i] = A[i, i-1]
        if i < n-1:
            c[i] = A[i, i+1]
    al = np.zeros(n)
    be = np.zeros(n)
    y = np.zeros(n)
    x = np.zeros(n)
    al[0] = b[0]
    for i in range(1, n):
        be[i] = a[i] / al[i-1]
        al[i] = b[i] - be[i] * c[i-1]
    y[0] = f[0]
    for i in range(1, n):
        y[i] = f[i] - be[i] * y[i-1]
    x[-1] = y[-1] / al[-1]
    for i in range(n-2, -1, -1):
        x[i] = (y[i] - c[i] * x[i+1]) / al[i]
    return x, al, be, y

t1 = time.time()
L = chol(M)
yy = f_sub(L, g)
xx = b_sub(L.T, yy)
t2 = time.time()

t3 = time.time()
xt = thomas(M.copy(), g.copy())
t4 = time.time()

t5 = time.time()
xn = LA.solve(M, g)
t6 = time.time()

print("Time Cholesky:", t2 - t1)
print("Time Thomas:", t4 - t3)
print("Time NumPy:", t6 - t5)
