import numpy as np


def dt_scale_3d(N, face_scale):
    dt_scale = 1.0 / (np.max(np.max(face_scale)) * N * N)
    return dt_scale


def curl_3d(Ux, Uy, Uz, Dr, Ds, Dt, rx, sx, tx, ry, sy, ty, rz, sz, tz):
    """compute local elemental physical spatial curl of (Ux, Uy, Uz)"""

    # compute local derivatives of Ux on reference tetrahedron
    ddr = Dr @ Ux
    dds = Ds @ Ux
    ddt = Dt @ Ux

    # increment curl components
    curly = rz * ddr + sz * dds + tz * ddt
    curlz = -(ry * ddr + sy * dds + ty * ddt)

    # compute local derivatives of Uy on reference tetrahedron
    ddr = Dr @ Uy
    dds = Ds @ Uy
    ddt = Dt @ Uy

    # increment curl components
    curlx = -(rz * ddr + sz * dds + tz * ddt)
    curlz += rx * ddr + sx * dds + tx * ddt

    # compute local derivatives of Uz on reference tetrahedron
    ddr = Dr @ Uz
    dds = Ds @ Uz
    ddt = Dt @ Uz

    # increment curl components
    curlx += ry * ddr + sy * dds + ty * ddt
    curly -= rx * ddr + sx * dds + tx * ddt

    return curlx, curly, curlz


def maxwell_rhs_3d(N, K, Hx, Hy, Hz, Ex, Ey, Ez, lift, face_scale, vmapP, vmapM, mapB, vmapB, nx, ny, nz, Dr, Ds, Dt, rx, sx, tx, ry, sy, ty, rz, sz, tz):
    """Evaluate RHS flux in 3D Maxwell equations"""

    # define constants
    Nfp = (N + 1) * (N + 2) // 2  # no. nodes on face
    Nfaces = 4  # no. faces per element

    # storage for field differences at faces

    # form field differences at faces
    vmapP = vmapP.astype(int) - 1
    vmapM = vmapM.astype(int) - 1
    vmapB = vmapB.astype(int) - 1

    Ex_shape = np.shape(Ex)
    Ey_shape = np.shape(Ey)
    Ez_shape = np.shape(Ez)

    dHx = (np.ravel(Hx, order='F')[vmapP] - np.ravel(Hx, order='F')[vmapM])
    dHy = (np.ravel(Hy, order='F')[vmapP] - np.ravel(Hy, order='F')[vmapM])
    dHz = (np.ravel(Hz, order='F')[vmapP] - np.ravel(Hz, order='F')[vmapM])
    dEx = (np.ravel(Ex, order='F')[vmapP] - np.ravel(Ex, order='F')[vmapM])
    dEy = (np.ravel(Ey, order='F')[vmapP] - np.ravel(Ey, order='F')[vmapM])
    dEz = (np.ravel(Ez, order='F')[vmapP] - np.ravel(Ez, order='F')[vmapM])

    Ex = np.ravel(Ex, order='F')
    Ey = np.ravel(Ey, order='F')
    Ez = np.ravel(Ez, order='F')

    # make boundary conditions all reflective (Ez+ = -Ez-)
    dHx[mapB] = 0
    dEx[mapB] = -2 * Ex[vmapB]
    dHy[mapB] = 0
    dEy[mapB] = -2 * Ey[vmapB]
    dHz[mapB] = 0
    dEz[mapB] = -2 * Ez[vmapB]

    dHx = dHx.reshape((Nfp*Nfaces, K), order='F')
    dHy = dHy.reshape((Nfp*Nfaces, K), order='F')
    dHz = dHz.reshape((Nfp*Nfaces, K), order='F')
    dEx = dEx.reshape((Nfp*Nfaces, K), order='F')
    dEy = dEy.reshape((Nfp*Nfaces, K), order='F')
    dEz = dEz.reshape((Nfp*Nfaces, K), order='F')
    Ex = Ex.reshape(Ex_shape, order='F')
    Ey = Ey.reshape(Ey_shape, order='F')
    Ez = Ez.reshape(Ez_shape, order='F')

    alpha = 1  # => full upwinding

    ndotdH = nx * dHx + ny * dHy + nz * dHz
    ndotdE = nx * dEx + ny * dEy + nz * dEz

    fluxHx = -ny * dEz + nz * dEy + alpha * (dHx - ndotdH * nx)
    fluxHy = -nz * dEx + nx * dEz + alpha * (dHy - ndotdH * ny)
    fluxHz = -nx * dEy + ny * dEx + alpha * (dHz - ndotdH * nz)

    fluxEx = ny * dHz - nz * dHy + alpha * (dEx - ndotdE * nx)
    fluxEy = nz * dHx - nx * dHz + alpha * (dEy - ndotdE * ny)
    fluxEz = nx * dHy - ny * dHx + alpha * (dEz - ndotdE * nz)

    # evaluate local spatial derivatives
    curlHx, curlHy, curlHz = curl_3d(Hx, Hy, Hz, Dr, Ds, Dt, rx, sx, tx, ry, sy, ty, rz, sz, tz)
    curlEx, curlEy, curlEz = curl_3d(Ex, Ey, Ez, Dr, Ds, Dt, rx, sx, tx, ry, sy, ty, rz, sz, tz)

    # calculate Maxwell's right hand side
    rhsHx = -curlEx + lift @ (face_scale * fluxHx / 2)
    rhsHy = -curlEy + lift @ (face_scale * fluxHy / 2)
    rhsHz = -curlEz + lift @ (face_scale * fluxHz / 2)

    rhsEx = curlHx + lift @ (face_scale * fluxEx / 2)
    rhsEy = curlHy + lift @ (face_scale * fluxEy / 2)
    rhsEz = curlHz + lift @ (face_scale * fluxEz / 2)

    return rhsHx, rhsHy, rhsHz, rhsEx, rhsEy, rhsEz


def maxwell_3d(N, K, Hx, Hy, Hz, Ex, Ey, Ez, FinalTime, lift, face_scale, vmapP, vmapM, mapB, vmapB, nx, ny, nz, Dr, Ds, Dt, rx, sx, tx, ry, sy, ty, rz, sz, tz):
    """Integrate 3D maxwell's until FinalTime starting with initial conditions
    Hx, Hy, Hz, Ex, Ey, Ez"""

    # definition of constants
    Np = (N + 1) * (N + 2) * (N + 3) // 6  # no. nodes in element
    rk4a = np.array([0.0,
                     -567301805773.0 / 1357537059087.0,
                     -2404267990393.0 / 2016746695238.0,
                     -3550918686646.0 / 2091501179385.0,
                     -1275806237668.0 / 842570457699.0])

    rk4b = np.array([1432997174477.0 / 9575080441755.0,
                     5161836677717.0 / 13612068292357.0,
                     1720146321549.0 / 2090206949498.0,
                     3134564353537.0 / 4481467310338.0,
                     2277821191437.0 / 14882151754819.0])

    # Runge-Kutta residual storage
    resHx = np.zeros((Np, K))
    resHy = np.zeros((Np, K))
    resHz = np.zeros((Np, K))
    resEx = np.zeros((Np, K))
    resEy = np.zeros((Np, K))
    resEz = np.zeros((Np, K))

    # compute time step size
    dt = dt_scale_3d(N, face_scale)  # TW: buggy

    # correct dt for integer # of time steps
    Ntsteps = int(np.ceil(FinalTime / dt))
    dt = FinalTime / Ntsteps

    time = 0
    tstep = 1

    while time < FinalTime:  # outer time step loop
        for i in range(0, 5):  # inner multi-stage Runge-Kutta loop
            # compute right hand side of TM-mode Maxwell's equations
            rhsHx, rhsHy, rhsHz, rhsEx, rhsEy, rhsEz = maxwell_rhs_3d(N, K, Hx, Hy, Hz, Ex, Ey, Ez, lift, face_scale, vmapP, vmapM, mapB, vmapB, nx, ny, nz, Dr, Ds, Dt, rx, sx, tx, ry, sy, ty, rz, sz, tz)

            # initiate, increment Runge-Kutta residuals and update fields
            resHx = rk4a[i] * resHx + dt * rhsHx
            Hx = Hx + rk4b[i] * resHx
            resHy = rk4a[i] * resHy + dt * rhsHy
            Hy = Hy + rk4b[i] * resHy
            resHz = rk4a[i] * resHz + dt * rhsHz
            Hz = Hz + rk4b[i] * resHz
            resEx = rk4a[i] * resEx + dt * rhsEx
            Ex = Ex + rk4b[i] * resEx
            resEy = rk4a[i] * resEy + dt * rhsEy
            Ey = Ey + rk4b[i] * resEy
            resEz = rk4a[i] * resEz + dt * rhsEz
            Ez = Ez + rk4b[i] * resEz

        time += dt  # Increment time
        tstep += 1
        print(tstep)

    return Hx, Hy, Hz, Ex, Ey, Ez