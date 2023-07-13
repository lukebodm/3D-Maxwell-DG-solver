import numpy as np

def mesh_reader3d(filename):
    '''
    Read in basic grid information to build grid
    gambit *.neu format is assumed

    :return:
    K: Number of elements in grid
    Nv: Number of vertices in the grid
    VX: row vector of x-coordinates for vertices in grid
    VY: row vector of x-coordinates for vertices in grid
    VZ: row vector of x-coordinates for vertices in grid
    EToV: 2D array, each row represents an element
          and the columns show the vertex number in the element
    '''
    f = open(filename, 'rt')

    # read intro
    for i in range(6):
        line = f.readline()

    # find number of nodes and number of elements
    dims = list(map(int, f.readline().split()))
    Nv = dims[0]
    K = dims[1]

    # skip lines
    for i in range(2):
        line = f.readline()

    # read node coordinates
    xyz = []
    for _ in range(Nv):
        coords = list(map(float, f.readline().split()))
        xyz.append(coords[1:4])
    xyz = list(map(list, zip(*xyz)))

    # create VX, VY, VZ vectors
    VX = np.array(xyz[0])
    VY = np.array(xyz[1])
    VZ = np.array(xyz[2])

    # skip lines
    for i in range(2):
        line = f.readline()

    # read element to node connectivity
    EToV = []
    for k in range(K):
        line = f.readline()
        tmpcon = list(map(float, line.split()))
        EToV.append(tmpcon[3:7])

    # subtract 1 to account for 0 indexing
    EToV = (np.array(EToV) - 1).astype(int)

    return Nv, VX, VY, VZ, K, EToV