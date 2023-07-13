import numpy as np

from polynomial_basis import jacobi_p, grad_jacobi_P


def grad_simplex_3dp(a, b, c, id, jd, kd):
    """ Return the derivatives of the modal basis (id,jd,kd) on the 3D simplex at (a,b,c)"""

    fa = jacobi_p(a, 0, 0, id)
    dfa = grad_jacobi_P(a, 0, 0, id)
    gb = jacobi_p(b, 2*id+1, 0, jd)
    dgb = grad_jacobi_P(b, 2*id+1, 0, jd)
    hc = jacobi_p(c, 2*(id+jd)+2, 0, kd)
    dhc = grad_jacobi_P(c, 2*(id+jd)+2, 0, kd)

    # calculate r derivative Vr
    Vr = dfa * (gb * hc)
    if id > 0:
        Vr *= (0.5 * (1 - b)) ** (id - 1)
    if id + jd > 0:
        Vr *= (0.5 * (1 - c)) ** (id + jd - 1)

    # calculate s derivative Vs
    Vs = 0.5 * (1 + a) * Vr
    tmp = dgb * ((0.5 * (1 - b)) ** id)
    if id > 0:
        tmp += (-0.5 * id) * (gb * ((0.5 * (1 - b)) ** (id - 1)))
    if id + jd > 0:
        tmp *= (0.5 * (1 - c)) ** (id + jd - 1)
    tmp = fa * (tmp * hc)
    Vs += tmp

    # calculate t derivative Vt
    Vt = 0.5 * (1 + a) * Vr + 0.5 * (1 + b) * tmp
    tmp = dhc * ((0.5 * (1 - c)) ** (id + jd))
    if id + jd > 0:
        tmp -= 0.5 * (id + jd) * (hc * ((0.5 * (1 - c)) ** (id + jd - 1)))
    tmp = fa * (gb * tmp)
    tmp *= (0.5 * (1 - b)) ** id
    Vt += tmp

    # normalize
    Vr *= 2 ** (2*id+jd+1.5)
    Vs *= 2 ** (2*id+jd+1.5)
    Vt *= 2 ** (2*id+jd+1.5)

    return Vr, Vs, Vt


def grad_vandermonde_3d(N, r, s, t):
    Vr = np.zeros((len(r), (N + 1) * (N + 2) * (N + 3) // 6))
    Vs = np.zeros((len(r), (N + 1) * (N + 2) * (N + 3) // 6))
    Vt = np.zeros((len(r), (N + 1) * (N + 2) * (N + 3) // 6))

    a, b, c = rst_to_abc(r, s, t)

    sk = 0
    for i in range(N + 1):
        for j in range(N - i + 1):
            for k in range(N - i - j + 1):
                Vr[:, sk], Vs[:, sk], Vt[:, sk] = grad_simplex_3dp(a, b, c, i, j, k)
                sk += 1

    return Vr, Vs, Vt

def diff_matrices_3d(N, r, s, t, V):
    Vr, Vs, Vt = grad_vandermonde_3d(N, r, s, t)
    Ds = np.matmul(Vs, np.linalg.inv(V))
    Dr = np.matmul(Vr, np.linalg.inv(V))
    Dt = np.matmul(Vt, np.linalg.inv(V))
    return Dr, Ds, Dt


def simplex_3d_polynomial(a, b, c, i, j, k):
    h1 = jacobi_p(a, 0, 0, i)
    h2 = jacobi_p(b, 2*i+1, 0, j)
    h3 = jacobi_p(c, 2*(i+j)+2, 0, k)

    P = 2 * np.sqrt(2) * h1 * h2 * ((1 - b) ** i) * h3 * ((1 - c) ** (i + j))
    return P

def simplex_2d_polynomial(a, b, i, j):
    """ Evaluate 2D orthonormal polynomial on simplex at (a,b) of order (i,j) """
    h1 = jacobi_p(a, 0, 0, i)
    h2 = jacobi_p(b, 2 * i + 1, 0, j)

    P = np.sqrt(2.0) * h1 * h2 * (1 - b) ** i
    return P

def rst_to_abc(r, s, t):
    """ transfer from (r,s,t) coordinates to (a,b,c) which are used to evaluate the
    jacobi polynomials in our orthonormal basis """
    Np = len(r)
    a = np.zeros(Np)
    b = np.zeros(Np)
    c = np.zeros(Np)

    for n in range(Np):
        if s[n] + t[n] != 0:
            a[n] = 2 * (1 + r[n]) / (-s[n] - t[n]) - 1
        else:
            a[n] = -1

        if t[n] != 1:
            b[n] = 2 * (1 + s[n]) / (1 - t[n]) - 1
        else:
            b[n] = -1

        c[n] = t[n]

    return a, b, c


def rs_to_ab(r, s):
    """ map from (r,s) coordinates on an element face to (a,b) which are used to evaluate the
    jacobi polynomials in our orthonormal basis """
    Np = len(r)
    a = np.zeros(Np)
    for n in range(Np):
        if s[n] != 1:
            a[n] = 2 * (1 + r[n]) / (1 - s[n]) - 1
        else:
            a[n] = -1
    b = s
    return a, b


def vandermonde_3d(N, r, s, t):
    # Initialize the 3D Vandermonde Matrix
    V = np.zeros((len(r), (N+1)*(N+2)*(N+3)//6))

    # Transfer to (a, b) coordinates
    a, b, c = rst_to_abc(r, s, t)

    # Build the Vandermonde matrix
    sk = 0

    for i in range(N + 1):
        for j in range(N - i + 1):
            for k in range(N - i - j + 1):
                V[:, sk] = simplex_3d_polynomial(a, b, c, i, j, k)
                sk += 1

    return V


def vandermonde_2d(N, r, s):
    """ create 2D vandermonde matrix to evaluate flux at faces of each element"""

    # initiate vandermonde matrix
    v = np.zeros((len(r), (N+1)*(N+2)//2))

    # Transfer to (a, b) coordinates
    a, b = rs_to_ab(r, s)

    # Build the Vandermonde matrix
    sk = 0
    for i in range(N+1):
        for j in range(N - i + 1):
            v[:, sk] = simplex_2d_polynomial(a, b, i, j)  # Assuming Simplex2DP function is defined
            sk += 1

    return v
