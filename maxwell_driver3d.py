import numpy as np
import vtk

from mesh_functions import mesh_reader3d
from plotting import plot_coordinates, plot_elements, plot_fields
from nodal_set import nodes_3d, xyz_to_rst
from elementwise_operators import vandermonde_3d, grad_vandermonde_3d, diff_matrices_3d
from lift import lift_3d
from geometric_factors import geometric_factors_3d, normals_3d
from connectivity import connectivity_mat_3D, build_maps_3d
from maxwell_solver import maxwell_3d


def find_face_nodes(r, s, t):
    """ return node indexes for nodes on each face of the standard tetrahedron"""
    fmask1 = np.where(np.abs(1 + t) < NODETOL)[0]
    fmask2 = np.where(np.abs(1 + s) < NODETOL)[0]
    fmask3 = np.where(np.abs(1 + r + s + t) < NODETOL)[0]
    fmask4 = np.where(np.abs(1 + r) < NODETOL)[0]
    Fmask = np.concatenate((fmask1, fmask2, fmask3, fmask4))
    #Fx = x[Fmask, :]
    #Fy = y[Fmask, :]
    #Fz = z[Fmask, :]

    return Fmask


def node_coordinates(EToV, VX, VY, VZ, r, s, t):
    """ returns x, y, and z arrays of coordinates of nodes from EToV and VX, VY, VZ, arrays"""

    # extract vertex numbers from elements
    va = EToV[:, 0].T
    vb = EToV[:, 1].T
    vc = EToV[:, 2].T
    vd = EToV[:, 3].T

    VX = VX.reshape(-1, 1)
    VY = VY.reshape(-1, 1)
    VZ = VZ.reshape(-1, 1)

    # map r, s, t from standard tetrahedron to x, y, z coordinates for each element
    x = (0.5 * (-(1 + r + s + t) * VX[va] + (1 + r) * VX[vb] + (1 + s) * VX[vc] + (1 + t) * VX[vd])).T
    y = (0.5 * (-(1 + r + s + t) * VY[va] + (1 + r) * VY[vb] + (1 + s) * VY[vc] + (1 + t) * VY[vd])).T
    z = (0.5 * (-(1 + r + s + t) * VZ[va] + (1 + r) * VZ[vb] + (1 + s) * VZ[vc] + (1 + t) * VZ[vd])).T

    return x, y, z


if __name__ == "__main__":

    # set polynomial order of approximation
    N = 6

    # Read in mesh
    Nv, VX, VY, VZ, K, EToV = mesh_reader3d('cubeK6.neu')

    # plot elements using Plotly
    ##plot_elements(VX, VY, VZ, K, EToV)

    # definition of constants
    Np = (N + 1) * (N + 2) * (N + 3) // 6  # no. nodes in element
    Nfp = (N + 1) * (N + 2) // 2  # no. nodes on face
    Nfaces = 4  # no. faces per element
    NODETOL = 1e-7  # tolerance to find face nodes

    # Compute nodal set
    x, y, z = nodes_3d(N)  # x,y,z generalized LGL coordinates on eq. tetrahedron
    r, s, t = xyz_to_rst(x, y, z) # r,s,t coordinates on standard tetrahedron, mapped from x, y, z
    ##plot_coordinates(np.array([r, s, t]).T, "Warped Coordinates on standard tetrahedron")

    # Build elementwise operators on reference element
    V = vandermonde_3d(N, r, s, t)
    invV = np.linalg.inv(V)
    MassMatrix = invV.T @ invV
    Dr, Ds, Dt = diff_matrices_3d(N, r, s, t, V)

    # build coordinates of all the nodes in each element
    x, y, z = node_coordinates(EToV, VX, VY, VZ, r, s, t)

    # find all the nodes that lie on each edge (to later compute surface integrals)
    face_mask = find_face_nodes(r, s, t)

    # create lift matrix to evaluate flux integral at boundary
    lift = lift_3d(N, r, s, t, face_mask, V) ## LIFT MATRIX IS SLIGHTLY DIFFERENT HERE

    rx, sx, tx, ry, sy, ty, rz, sz, tz, J = geometric_factors_3d(x, y, z, Dr, Ds, Dt)

    nx, ny, nz, sJ = normals_3d(x, y, z, Dr, Ds, Dt, face_mask, Nfp, K)
    face_scale = sJ / J[face_mask, :]

    # build connectivity matrices
    EToE, EToF = connectivity_mat_3D(EToV)

    # build connectivity maps
    vmapM, vmapP, vmapB, mapB = build_maps_3d(N, K, face_mask, EToE, EToF, x, y, z)

    Vr, Vs, Vt = grad_vandermonde_3d(N, r, s, t)

    Drw = np.matmul(np.matmul(V, Vr.T), np.linalg.inv(np.matmul(V, V.T)))
    Dsw = np.dot(np.dot(V, Vs.T), np.linalg.inv(np.dot(V, V.T)))
    Dtw = np.dot(np.dot(V, Vt.T), np.linalg.inv(np.dot(V, V.T)))

    # Set initial conditions
    mmode = 1
    nmode = 1

    # Use TM mode Maxwell's initial condition
    Hx = np.zeros((Np, K))
    Hy = np.zeros((Np, K))
    Hz = np.zeros((Np, K))
    Ex = np.zeros((Np, K))
    Ey = np.zeros((Np, K))

    # Ez = exp(-20*(x.^2 + y.^2))
    xmode = 1
    ymode = 1
    Ez = np.sin(xmode * np.pi * x) * np.sin(ymode * np.pi * y)

    # Solve Problem
    FinalTime = 10

    Hx, Hy, Hz, Ex, Ey, Ez = maxwell_3d(N, K, Hx, Hy, Hz, Ex, Ey, Ez,
                                        FinalTime, lift, face_scale,
                                        vmapP, vmapM, mapB, vmapB,
                                        nx, ny, nz, Dr, Ds, Dt,
                                        rx, sx, tx, ry, sy, ty, rz, sz, tz)

    export_to_paraview(Hx, x, y, z)