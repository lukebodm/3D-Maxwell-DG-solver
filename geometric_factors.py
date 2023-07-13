import numpy as np


def normals_3d(x, y, z, Dr, Ds, Dt, Fmask, Nfp, K):
    """compute outward pointing normals at elements faces as well as surface Jacobians"""
    rx, sx, tx, ry, sy, ty, rz, sz, tz, J = geometric_factors_3d(x, y, z, Dr, Ds, Dt)

    # interpolate geometric factors to face nodes
    frx = rx[Fmask, :]
    fsx = sx[Fmask, :]
    ftx = tx[Fmask, :]
    fry = ry[Fmask, :]
    fsy = sy[Fmask, :]
    fty = ty[Fmask, :]
    frz = rz[Fmask, :]
    fsz = sz[Fmask, :]
    ftz = tz[Fmask, :]

    # build normal vectors
    nx = np.zeros((4 * Nfp, K))
    ny = np.zeros((4 * Nfp, K))
    nz = np.zeros((4 * Nfp, K))

    # create vectors of indices of each face
    fid1 = np.arange(1, Nfp + 1)
    fid2 = np.arange(Nfp + 1, 2 * Nfp + 1)
    fid3 = np.arange(2 * Nfp + 1, 3 * Nfp + 1)
    fid4 = np.arange(3 * Nfp + 1, 4 * Nfp + 1)

    # face 1
    nx[fid1 - 1, :] = -ftx[fid1 - 1, :]
    ny[fid1 - 1, :] = -fty[fid1 - 1, :]
    nz[fid1 - 1, :] = -ftz[fid1 - 1, :]

    # face 2
    nx[fid2 - 1, :] = -fsx[fid2 - 1, :]
    ny[fid2 - 1, :] = -fsy[fid2 - 1, :]
    nz[fid2 - 1, :] = -fsz[fid2 - 1, :]

    # face 3
    nx[fid3 - 1, :] = frx[fid3 - 1, :] + fsx[fid3 - 1, :] + ftx[fid3 - 1, :]
    ny[fid3 - 1, :] = fry[fid3 - 1, :] + fsy[fid3 - 1, :] + fty[fid3 - 1, :]
    nz[fid3 - 1, :] = frz[fid3 - 1, :] + fsz[fid3 - 1, :] + ftz[fid3 - 1, :]

    # face 4
    nx[fid4 - 1, :] = -frx[fid4 - 1, :]
    ny[fid4 - 1, :] = -fry[fid4 - 1, :]
    nz[fid4 - 1, :] = -frz[fid4 - 1, :]

    # find surface Jacobian
    sJ = np.sqrt(nx * nx + ny * ny + nz * nz)
    nx /= sJ
    ny /= sJ
    nz /= sJ
    sJ *= J[Fmask, :]

    return nx, ny, nz, sJ


def geometric_factors_3d(x, y, z, Dr, Ds, Dt):
    """Compute the metric elements for the local mappings of the elements"""

    # find jacobian of mapping
    xr = np.dot(Dr, x)
    xs = np.dot(Ds, x)
    xt = np.dot(Dt, x)
    yr = np.dot(Dr, y)
    ys = np.dot(Ds, y)
    yt = np.dot(Dt, y)
    zr = np.dot(Dr, z)
    zs = np.dot(Ds, z)
    zt = np.dot(Dt, z)

    J = xr * (ys * zt - zs * yt) - yr * (xs * zt - zs * xt) + zr * (xs * yt - ys * xt)

    # compute the metric constants
    rx = (ys * zt - zs * yt) / J
    ry = -(xs * zt - zs * xt) / J
    rz = (xs * yt - ys * xt) / J
    sx = -(yr * zt - zr * yt) / J
    sy = (xr * zt - zr * xt) / J
    sz = -(xr * yt - yr * xt) / J
    tx = (yr * zs - zr * ys) / J
    ty = -(xr * zs - zr * xs) / J
    tz = (xr * ys - yr * xs) / J

    return rx, sx, tx, ry, sy, ty, rz, sz, tz, J
