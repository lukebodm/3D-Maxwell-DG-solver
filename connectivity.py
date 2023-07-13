import numpy as np


def build_maps_3d(N, K, face_mask, EToE, EToF, x, y, z):

    # definition of constants
    Np = (N + 1) * (N + 2) * (N + 3) // 6  # no. nodes in element
    Nfp = (N + 1) * (N + 2) // 2  # no. nodes on face
    Nfaces = 4  # no. faces per element
    NODETOL = 1e-7  # tolerance to find face nodes

    # declare map matrices
    nodeids = np.arange(1, (K * Np) + 1).reshape(Np, K, order='F')
    vmapM = np.zeros((Nfp, Nfaces, K))  # Vector of global nodal numbers at faces for interior values, u−
    vmapP = np.zeros((Nfp, Nfaces, K))  # Vector of global nodal numbers at faces for exterior values, u+

    # reshape face_mask
    face_mask = face_mask.reshape(4, -1).T

    for k1 in range(K):
        for f1 in range(Nfaces):
            vmapM[:, f1, k1] = nodeids[face_mask[:, f1], k1]

    tmp = np.ones((1, Nfp))
    for k1 in range(K):
        for f1 in range(Nfaces):
            k2 = EToE[k1, f1]
            f2 = EToF[k1, f1]
            vidM = vmapM[:, f1, k1].astype(int)
            vidP = vmapM[:, f2, k2].astype(int)

            xM = (np.ravel(x, order='F')[vidM - 1][:, np.newaxis] @ tmp)
            yM = (np.ravel(y, order='F')[vidM - 1][:, np.newaxis] @ tmp)
            zM = (np.ravel(z, order='F')[vidM - 1][:, np.newaxis] @ tmp)
            xP = (np.ravel(x, order='F')[vidP - 1][:, np.newaxis] @ tmp)
            yP = (np.ravel(y, order='F')[vidP - 1][:, np.newaxis] @ tmp)
            zP = (np.ravel(z, order='F')[vidP - 1][:, np.newaxis] @ tmp)

            D = (xM - xP.T)**2 + (yM - yP.T)**2 + (zM - zP.T)**2
            idM, idP = np.where(np.abs(D) < NODETOL)
            vmapP[idM, f1, k1] = vmapM[idP, f2, k2]

    vmapP = vmapP.reshape(-1, order='F')
    vmapM = vmapM.reshape(-1, order='F')

    mapB = np.where(vmapP == vmapM)[0]
    vmapB = vmapM[mapB]

    return vmapM, vmapP, vmapB, mapB


def connectivity_mat_3D(EToV):
    """tetrahedral face connect algorithm due to Toby Isaac"""

    Nfaces = 4
    K = EToV.shape[0]
    Nnodes = np.max(EToV)

    # create list of all faces
    fnodes = np.vstack((EToV[:, [0, 1, 2]],
                        EToV[:, [0, 1, 3]],
                        EToV[:, [1, 2, 3]],
                        EToV[:, [0, 2, 3]]))
    fnodes = np.sort(fnodes, axis=1) - 1

    # set up default element to element and element to faces connectivity
    EToE = np.tile(np.arange(1, K+1)[:, np.newaxis], (1, Nfaces))
    EToF = np.tile(np.arange(1, Nfaces+1), (K, 1))

    # uniquely number each set of three faces by their node numbers
    id = fnodes[:, 0] * Nnodes * Nnodes + fnodes[:, 1] * Nnodes + fnodes[:, 2] + 1
    spNodeToNode = np.column_stack((id, np.arange(1, Nfaces*K+1), np.ravel(EToE, order='F'), np.ravel(EToF, order='F')))

    # Now we sort by global face number.
    sorted_spNode = np.array(sorted(spNodeToNode, key=lambda x: (x[0], x[1])))

    # find matches in the sorted face list
    matches = np.where(sorted_spNode[:-1, 0] == sorted_spNode[1:, 0])[0]

    # make links reflexive
    matchL = np.vstack((sorted_spNode[matches], sorted_spNode[matches + 1]))
    matchR = np.vstack((sorted_spNode[matches + 1], sorted_spNode[matches]))

    # insert matches
    EToE_tmp = np.ravel(EToE, order='F') - 1
    EToF_tmp = np.ravel(EToF, order='F') - 1

    EToE_tmp[matchL[:, 1] - 1] = (matchR[:, 2] - 1)
    EToF_tmp[matchL[:, 1] - 1] = (matchR[:, 3] - 1)

    EToE = EToE_tmp.reshape(EToE.shape, order='F') #+ 1
    EToF = EToF_tmp.reshape(EToF.shape, order='F') #+ 1

    return EToE, EToF
