import plotly.graph_objects as go
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots
import numpy as np
from mesh_functions import mesh_reader3d

def plot_elements(VX, VY, VZ, K, EToV):
    # Define the subplot arrangement
    rows = 2
    cols = 3

    # Create the subplots
    fig = make_subplots(rows=rows, cols=cols, subplot_titles=[f'Element {k}' for k in range(K)], specs=[[{'type': 'scene'}]*cols]*rows)

    # Iterate over each element and create subplots
    for k in range(K):
        element_vertices = EToV[k, :]  # Subtract 1 to account for 0-based indexing
        x = VX[element_vertices]
        y = VY[element_vertices]
        z = VZ[element_vertices]

        # Define the face indices
        faces = [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]]

        # Create the Mesh3d trace for the tetrahedron
        mesh_trace = go.Mesh3d(
            x=x,
            y=y,
            z=z,
            i=[face[0] for face in faces],
            j=[face[1] for face in faces],
            k=[face[2] for face in faces],
            color='lightblue',
            opacity=0.6
        )

        # Add the trace to the corresponding subplot
        row = k // cols + 1
        col = k % cols + 1
        fig.add_trace(mesh_trace, row=row, col=col)

    # Set the layout for all subplots
    fig.update_layout(scene=dict(xaxis=dict(range=[-1, 1]),
                                 yaxis=dict(range=[-1, 1]),
                                 zaxis=dict(range=[-1, 1])),
                      title='All Elements in Domain')

    # Show the figure
    fig.show()


def plot_eq_tetra(v1, v2 ,v3, v4, t1, t2):

    # create figure and axis
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # plot the vertices
    ax.scatter(v1[0], v1[1], v1[2], color='red', label='v1')
    ax.scatter(v2[0], v2[1], v2[2], color='green', label='v2')
    ax.scatter(v3[0], v3[1], v3[2], color='blue', label='v3')
    ax.scatter(v4[0], v4[1], v4[2], color='purple', label='v4')

    # plot the edges connecting the vertices
    lines = [
        [v1, v2],
        [v1, v3],
        [v1, v4],
        [v2, v3],
        [v2, v4],
        [v3, v4]
    ]
    for line in lines:
        ax.plot(*zip(line[0], line[1]), color='black')

    # Calculate the center points of each face
    face_centers = [(v1 + v2 + v3) / 3,  (v1 + v2 + v4) / 3, (v2 + v3 + v4) / 3, (v1 + v3 + v4) / 3]

    # scale down tangent vectors
    t1 = t1/2
    t2 = t2/2
    # plot the vectors t1 and t2 with their origins at the center of the faces
    for i in range(4):
        ax.quiver(face_centers[i][0], face_centers[i][1], face_centers[i][2],
                  t1[i, 0], t1[i, 1], t1[i, 2], color='orange', label=f't1')
        ax.quiver(face_centers[i][0], face_centers[i][1], face_centers[i][2],
                  t2[i, 0], t2[i, 1], t2[i, 2], color='purple', label='t2')

    # set axes limits
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)

    # label axes
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # set title
    ax.set_title('Equilateral tetrahedron from nodes_3d()')  # Set the title

    plt.legend()
    plt.show()


def plot_coordinates(xyz, title):
    """ plots a matrix of x y and z coordinates as dots"""
    x = xyz[:, 0]
    y = xyz[:, 1]
    z = xyz[:, 2]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z, color='red', label='v1')

    # set axes limits
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)

    # label axes
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # set title
    ax.set_title(title)  # Set the title

    plt.legend()
    plt.show()


def plot_coordinates_together(xyz, title, subplot_pos, fig):
    """ plots multiple matrices of 3 coordinates as dots on
    separate subplots. Requires figure object"""

    # extract coordinates
    x = xyz[:, 0]
    y = xyz[:, 1]
    z = xyz[:, 2]

    # create axis and plot
    ax = fig.add_subplot(subplot_pos, projection='3d')
    ax.scatter(x, y, z)

    # set axes limits
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)

    # label axes
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # set title
    ax.set_title(title)

def plot_fields(Hx, Hy, Hz, Ex, Ey, Ez, x, y, z):

    # Plotting magnetic field components
    fig = plt.figure()

    ax1 = fig.add_subplot(231, projection='3d')
    ax1.set_title('Hx')
    ax1.quiver(x, y, z, Hx, np.zeros_like(Hx), np.zeros_like(Hx), length=1, normalize=True)

    ax2 = fig.add_subplot(232, projection='3d')
    ax2.set_title('Hy')
    ax2.quiver(x, y, z, np.zeros_like(Hy), Hy, np.zeros_like(Hy), length=1, normalize=True)

    ax3 = fig.add_subplot(233, projection='3d')
    ax3.set_title('Hz')
    ax3.quiver(x, y, z, np.zeros_like(Hz), np.zeros_like(Hz), Hz, length=1, normalize=True)

    # Plotting electric field components
    ax4 = fig.add_subplot(234, projection='3d')
    ax4.set_title('Ex')
    ax4.quiver(x, y, z, Ex, np.zeros_like(Ex), np.zeros_like(Ex), length=1, normalize=True)

    ax5 = fig.add_subplot(235, projection='3d')
    ax5.set_title('Ey')
    ax5.quiver(x, y, z, np.zeros_like(Ey), Ey, np.zeros_like(Ey), length=1, normalize=True)

    ax6 = fig.add_subplot(236, projection='3d')
    ax6.set_title('Ez')
    ax6.quiver(x, y, z, np.zeros_like(Ez), np.zeros_like(Ez), Ez, length=1, normalize=True)

    plt.show()

