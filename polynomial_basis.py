import numpy as np

from scipy.special import gamma
from scipy.linalg import eig


def jacobi_p(x, alpha, beta, N):
    # Turn points into row if needed.
    xp = x

    # PL will carry our values for the jacobi polynomial
    PL = np.zeros((N+1, len(x)))

    # initialize values P_0(x)
    gamma0 = 2 ** (alpha + beta + 1) / (alpha + beta + 1) * gamma(alpha + 1) * \
             gamma(beta + 1) / gamma(alpha + beta + 1)
    PL[0, :] = 1.0 / np.sqrt(gamma0)

    # return if N = 0
    if N == 0:
        P = PL[0, :]
        return P

    # initialize value P_1(x)
    gamma1 = (alpha + 1) * (beta + 1) / (alpha + beta + 3) * gamma0
    PL[1, :] = ((alpha + beta + 2) * xp / 2 + (alpha - beta) / 2) / np.sqrt(gamma1)

    # return if N = 1
    if N == 1:
        P = PL[N, :]
        return P

    # Repeat value in recurrence.
    a_old = 2 / (2 + alpha + beta) * np.sqrt((alpha + 1) * (beta + 1) / (alpha + beta + 3))

    # Forward recurrence using the symmetry of the recurrence.
    for i in range(1, N):
        h1 = 2 * i + alpha + beta
        a_new = 2 / (h1 + 2) * np.sqrt((i + 1) * (i + 1 + alpha + beta) * (i + 1 + alpha) *
                                       (i + 1 + beta) / (h1 + 1) / (h1 + 3))
        b_new = -(alpha**2 - beta**2) / h1 / (h1 + 2)
        PL[i + 1, :] = 1 / a_new * (-a_old * PL[i-1, :] + (xp - b_new) * PL[i, :])
        a_old = a_new

    #P = np.reshape(PL[N, :], (np.shape(PL[N, :])[0], 1)).T
    P = PL[N, :]
    return P


def jacobi_GQ_points(alpha, beta, N):
    """Compute the N'th order Gauss quadrature points, x,
    and weights, w, associated with the Jacobi polynomial
    of type (alpha, beta) > -1 ( <> -0.5)."""

    x = np.zeros(N+1)
    w = np.zeros(N+1)

    if N == 0:
        x[0] = -(alpha-beta)/(alpha+beta+2)
        w[0] = 2
        return x, w

    # Form symmetric matrix from recurrence
    h1 = 2 * (np.arange(N+1)) + alpha + beta
    J = np.diag(-1/2 * (alpha**2 - beta**2) / (h1 + 2) / h1) + \
        np.diag(2 / (h1[0:N] + 2) * np.sqrt((np.arange(1, N+1)) * ((np.arange(1, N+1)) + alpha + beta) * \
        ((np.arange(1, N+1)) + alpha) * ((np.arange(1, N+1)) + beta) / \
        (h1[0:N] + 1) / (h1[0:N] + 3)), 1)

    if alpha + beta < 10 * np.finfo(float).eps:
        J[0, 0] = 0.0

    J = J + J.T

    # Compute quadrature by eigenvalue solve
    D, V = eig(J)
    sorted_indices = np.argsort(D)
    D = D[sorted_indices]
    V = V[:, sorted_indices]
    x = np.real(D)

    w = np.real((V[0, :]**2) * 2**(alpha + beta + 1) / (alpha + beta + 1) * \
        gamma(alpha + 1) * gamma(beta + 1) / gamma(alpha + beta + 1))

    return x, w


def jacobi_GL_points(alpha, beta, N):
    """ Compute the N'th order Gauss Lobatto quadrature points, x,
    associated with the Jacobi polynomial of type (alpha, beta) > -1 ( <> -0.5)."""

    x = np.zeros(N+1)
    if N == 1:
        x[0] = -1.0
        x[1] = 1.0
        return x

    xint, w = jacobi_GQ_points(alpha+1, beta+1, N-2)
    x[0] = -1
    x[1:N] = xint
    x[N] = 1

    return x


def grad_jacobi_P(r, alpha, beta, N):
    dP = np.zeros(len(r))
    if N == 0:
        dP[:] = 0.0
    else:
        dP = np.sqrt(N * (N + alpha + beta + 1)) * jacobi_p(r, alpha + 1, beta + 1, N - 1)
    return dP