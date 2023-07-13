import numpy as np
from polynomial_basis import jacobi_GL_points
from plotting import plot_eq_tetra, plot_coordinates, plot_coordinates_together
import matplotlib.pyplot as plt

def equidistributed_nodes_3d(N):
    """Compute the equidistributed node coordinates on the reference tetrahedron"""

    # Total number of nodes
    Np = (N+1)*(N+2)*(N+3)//6

    # Create equidistributed nodes on the equilateral triangle
    x = np.zeros(Np)
    y = np.zeros(Np)
    z = np.zeros(Np)

    sk = 0
    for n in range(1, N+2):
        for m in range(1, N+3-n):
            for q in range(1, N+4-n-m):
                x[sk] = -1 + (q-1)*2/N
                y[sk] = -1 + (m-1)*2/N
                z[sk] = -1 + (n-1)*2/N
                sk += 1

    return x, y, z


def eval_warp(p, xnodes, xout):
    """Compute one-dimensional edge warping function"""

    warp = np.zeros_like(xout)

    xeq = np.zeros(p+1)
    for i in range(p+1):
        xeq[i] = -1 + 2*(p-i)/p

    for i in range(1, p + 2):
        d = (xnodes[i - 1] - xeq[i - 1])

        for j in range(2, p + 1):
            if i != j:
                d *= (xout - xeq[j - 1]) / (xeq[i - 1] - xeq[j - 1])

        if i != 1:
            d = -d / (xeq[i - 1] - xeq[0])

        if i != (p + 1):
            d = d / (xeq[i - 1] - xeq[p])

        warp += d

    return warp


def eval_shift(p, pval, L1, L2, L3):
    """Compute two-dimensional Warp & Blend transform"""

    # 1) Compute Gauss-Lobatto-Legendre node distribution
    gaussX = -jacobi_GL_points(0, 0, p)

    # 2) Compute blending function at each node for each edge
    blend1 = L2 * L3
    blend2 = L1 * L3
    blend3 = L1 * L2

    # 3) Amount of warp for each node, for each edge
    warpfactor1 = 4 * eval_warp(p, gaussX, L3 - L2)
    warpfactor2 = 4 * eval_warp(p, gaussX, L1 - L3)
    warpfactor3 = 4 * eval_warp(p, gaussX, L2 - L1)

    # 4) Combine blend & warp
    warp1 = blend1 * warpfactor1 * (1 + (pval * L1) ** 2)
    warp2 = blend2 * warpfactor2 * (1 + (pval * L2) ** 2)
    warp3 = blend3 * warpfactor3 * (1 + (pval * L3) ** 2)

    # 5) Evaluate shift in equilateral triangle
    dx = 1 * warp1 + np.cos(2 * np.pi / 3) * warp2 + np.cos(4 * np.pi / 3) * warp3
    dy = 0 * warp1 + np.sin(2 * np.pi / 3) * warp2 + np.sin(4 * np.pi / 3) * warp3

    return dx, dy


def warp_shift_face_3d(p, pval, L2, L3, L4):
    """ compute warp factor used in creating 3D Warp & Blend nodes """

    dtan1, dtan2 = eval_shift(p, pval, L2, L3, L4)
    warpx = dtan1
    warpy = dtan2

    return warpx, warpy


def barycentric_coordinates(r, s, t):
    """ return the barycentric coordinates for a given r, s, t"""
    L1 = (1+t)/2
    L2 = (1+s)/2
    L3 = -(1+r+s+t)/2
    L4 = (1+r)/2
    return L1.reshape(-1, 1), L2.reshape(-1, 1), L3.reshape(-1, 1), L4.reshape(-1, 1)


def get_alpha_value(N):
    """ function that stores optimal blending parameter as calculated in
    Hesthaven, Warburton - Nodal DG methods """

    alpha_store = [0, 0, 0, 0.1002, 1.1332, 1.5608, 1.3413, 1.2577, 1.1603,
                   1.10153, 0.6080, 0.4523, 0.8856, 0.8717, 0.9655]

    # If N is greater than 15, alpha = 1 is a good enough approximation.
    if N <= 15:
        alpha = alpha_store[N-1]
    else:
        alpha = 1.0
    return alpha


def eq_tetra_vertices():
    """ Creates the 3D coordinates for the vertices of an
    equilateral tetrahedron """
    v1 = np.array([-1, -1/np.sqrt(3), -1/np.sqrt(6)]).reshape(1, 3)
    v2 = np.array([1, -1/np.sqrt(3), -1/np.sqrt(6)]).reshape(1, 3)
    v3 = np.array([0, 2/np.sqrt(3), -1/np.sqrt(6)]).reshape(1, 3)
    v4 = np.array([0, 0, 3/np.sqrt(6)]).reshape(1, 3)
    return v1, v2, v3, v4


def face_tangents(v1, v2, v3, v4):
    """ find the two orthogonal tangent vectors to each face """

    # create t1 and t2 vectors
    t1 = np.zeros((4, 3))
    t2 = np.zeros((4, 3))

    # compute tangent vectors
    t1[0, :] = v2 - v1
    t1[1, :] = v2 - v1
    t1[2, :] = v3 - v2
    t1[3, :] = v3 - v1
    t2[0, :] = v3 - 0.5*(v1 + v2)
    t2[1, :] = v4 - 0.5*(v1 + v2)
    t2[2, :] = v4 - 0.5*(v2 + v3)
    t2[3, :] = v4 - 0.5*(v1 + v3)

    # Normalize tangents
    for n in range(4):
        t1[n, :] = t1[n, :] / np.linalg.norm(t1[n, :])
        t2[n, :] = t2[n, :] / np.linalg.norm(t2[n, :])
    return t1, t2


def corresponding_lambdas(face_number, L1, L2, L3, L4):
    """ select the correct barycentric coordinates for each face"""
    if face_number == 0:
        return L1, L2, L3, L4
    elif face_number == 1:
        return L2, L1, L3, L4
    elif face_number == 2:
        return L3, L1, L4, L2
    elif face_number == 3:
        return L4, L1, L3, L2


def warp_and_blend(N, xyz, t1, t2, L1, L2, L3, L4):
    """ warp and blend the equidistant nodes on an equilateral tetrahedron
    to the generalization of Legendre-Gauss_Lobatto points"""

    # Choose optimized blending parameter
    alpha = get_alpha_value(N)

    # declare shift variable that will add to xyz
    shift = np.zeros_like(xyz)

    # tolerance to avoid dividing by zero
    tol = 1e-10

    # calculate warp amount for each face
    for face in range(4):
        La, Lb, Lc, Ld = corresponding_lambdas(face, L1, L2, L3, L4)

        # Compute warp tangential to the face
        warp1, warp2 = warp_shift_face_3d(N, alpha, Lb, Lc, Ld)

        # Compute volume blending
        blend = Lb * Lc * Ld

        # Modify linear blend
        denominator = (Lb + 0.5 * La) * (Lc + 0.5 * La) * (Ld + 0.5 * La)
        ids = np.where(denominator > tol)[0]
        blend[ids] = (1 + (alpha * La[ids]) ** 2) * blend[ids] / denominator[ids]

        # Compute warp & blend
        shift += (blend * warp1) @ np.reshape(t1[face, :], (1,3)) + (blend * warp2) @ np.reshape(t2[face, :], (1, 3))

        # Fix face warp
        ids = np.where((La < tol) & (np.array(Lb > tol, dtype=int) + np.array(Lc > tol, dtype=int) + np.array(Ld > tol, dtype=int) < 3))[0]

        shift[ids, :] = warp1[ids, :] * t1[face, :] + warp2[ids, :] * t2[face, :]

    # add warp and bend shift to XYZ
    xyz += shift
    return xyz


def nodes_3d(N):
    """
    Compute Warp & Blend nodes
    Input: p = polynomial order of interpolant
    Output: x, y, z = vectors of node coordinates in an equilateral tetrahedron
    """

    # Create equidistributed nodes
    r, s, t = equidistributed_nodes_3d(N)

    # get barycentric coordinates (lambdas) in terms of r,s,t
    L1, L2, L3, L4 = barycentric_coordinates(r, s, t)

    # get vertices of equilateral tetrahedron
    v1, v2, v3, v4 = eq_tetra_vertices()

    # find tangent vectors at each face
    t1, t2 = face_tangents(v1, v2, v3, v4)

    # Plot the points
    ##plot_eq_tetra(v1.T, v2.T, v3.T, v4.T, t1, t2)

    # Form undeformed coordinates
    xyz = L3 @ v1 + L4 @ v2 + L2 @ v3 + L1 @ v4

    # Create figure to plot undeformed and deformed coordinates
    ##fig = plt.figure()

    # Call plot_coordinates for the first subplot with xyz data
    ##plot_coordinates_together(xyz, "Undeformed coordinates from nodes_3d()", 121, fig)

    # Warp and blend for each face (accumulated in shiftXYZ)
    xyz = warp_and_blend(N, xyz, t1, t2, L1, L2, L3, L4)

    # Call plot_coordinates for the second subplot with rst data
    ##plot_coordinates_together(xyz, "Deformed coordinates from nodes_3d()", 122, fig)

    # Show the figure
    ##plt.show()

    x = xyz[:, 0]
    y = xyz[:, 1]
    z = xyz[:, 2]

    return x, y, z


def xyz_to_rst(x, y, z):
    """ map x,y,z to the standard tetrahedron """

    # define vertices of standard tetrahedron
    v1 = np.array([-1, -1 / np.sqrt(3), -1 / np.sqrt(6)])
    v2 = np.array([1, -1 / np.sqrt(3), -1 / np.sqrt(6)])
    v3 = np.array([0, 2 / np.sqrt(3), -1 / np.sqrt(6)])
    v4 = np.array([0, 0 / np.sqrt(3), 3 / np.sqrt(6)])

    # subtract the average coordinates of the vertices v1, v2, v3, and v4
    # (with appropriate adjustments) from the given X, Y, and Z coordinates.
    # This step aligns the coordinates with the reference tetrahedron.
    rhs = (np.array([x, y, z]).T - np.array([0.5 * (v2 + v3 + v4 - v1)])).T

    # solve matrix equation for mapping on pg 410
    A = np.column_stack(
        [0.5 * (v2 - v1), 0.5 * (v3 - v1), 0.5 * (v4 - v1)])

    RST = np.linalg.solve(A, rhs)

    r = RST[0, :].T
    s = RST[1, :].T
    t = RST[2, :].T

    return r, s, t
