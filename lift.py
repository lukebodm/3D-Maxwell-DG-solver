import numpy as np
from elementwise_operators import vandermonde_2d


def lift_3d(N, r, s, t, face_mask, V):
    """ Compute 3D surface to volume lift operator used in DG formulation """

    # definition of constants
    Np = (N + 1) * (N + 2) * (N + 3) // 6  # no. nodes in element
    Nfp = (N + 1) * (N + 2) // 2  # no. nodes on face
    Nfaces = 4

    # rearrange face_mask
    face_mask = face_mask.reshape(4, -1).T

    # initiate epsilon matrix
    E_matrix = np.zeros((Np, Nfaces * Nfp))

    for face in range(1, Nfaces + 1):
        # get the nodes on the specific face
        if face == 1:
            faceR = r[face_mask[:, 0]]
            faceS = s[face_mask[:, 0]]
        elif face == 2:
            faceR = r[face_mask[:, 1]]
            faceS = t[face_mask[:, 1]]
        elif face == 3:
            faceR = s[face_mask[:, 2]]
            faceS = t[face_mask[:, 2]]
        elif face == 4:
            faceR = s[face_mask[:, 3]]
            faceS = t[face_mask[:, 3]]

        v_face = vandermonde_2d(N, faceR, faceS)
        mass_face = np.linalg.inv(v_face @ v_face.T)

        idr = face_mask[:, face - 1]
        idc = np.arange((face - 1) * Nfp, face * Nfp)

        E_matrix[idr[:, np.newaxis], idc] += mass_face

    lift = V @ (V.T @ E_matrix)
    return lift
