import numpy as np
import sys
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from construct_sdp import construct_sdp

from solve_sdp import solve_sdp
from solve_smaller_sdp import solve_smaller_sdp
from solve_problem_so3 import solve_problem_so3
from staircase import solve_problem

from scipy.linalg import kron
from scipy.io import savemat

import time

do_solve_sdp = False # cvxpy SCS SDP solver
do_solve_smaller_sdp = True # cvxpy SCS SDP solver
do_solve_manopt_SO3 = True # pymanopt SO3 manifold solver

# !! must use matlab version !!
do_solve_staircase = False # pymanopt staircase Stiefel manifold solver !! must use matlba version !!

def construct_r_from_R(R):
    n = R.shape[0] // 3  # Number of 3x3 blocks
    d = 3  # Dimension of each block
    # Initialize an empty list to store the vectorized blocks
    vec_blocks = []
    for i in range(n):
        # Extract the i-th 3x3 block
        Ri = R[i*d:(i+1)*d, :]
        # Vectorize the block in column-major order and append to the list
        vec_blocks.append(Ri.reshape(-1))
    # Stack the vectorized blocks vertically to form the vector r
    r = np.concatenate(vec_blocks)
    return r

def construct_R_from_r(r, n):
    d = 3  # Dimension of each block
    R = np.zeros((d * n, d))  # Initialize R with the correct shape
    # Split the vector r into n blocks of size 9 and reshape each block to 3x3
    for i in range(n):
        ri = r[i*d*d:(i+1)*d*d]
        Ri = ri.reshape(d, d)
        R[i*d:(i+1)*d, :] = Ri
    return R

# undo Q = kron(Q_tild, Id)
def retrieve_Q_tild_from_Q(Q):
    d = 3  # Dimension of the identity matrix I3
    n = Q.shape[0] // d  # Number of blocks

    # Initialize an empty matrix Q
    Q_tild = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            # Extract the (i, j) block
            Q_tild[i, j] = Q[i*d, j*d]
    return Q_tild

def plot_poses(vertices, scale_fraction=0.1):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Collect all positions to determine axis limits
    positions = np.array([v[:3] for v in vertices])
    x_limits = [np.min(positions[:, 0]), np.max(positions[:, 0])]
    y_limits = [np.min(positions[:, 1]), np.max(positions[:, 1])]
    z_limits = [np.min(positions[:, 2]), np.max(positions[:, 2])]
    max_range = max(x_limits[1] - x_limits[0], y_limits[1] - y_limits[0], z_limits[1] - z_limits[0])
    scale = max_range * scale_fraction

    # Set equal axes limits
    mid_x = np.mean(x_limits)
    mid_y = np.mean(y_limits)
    mid_z = np.mean(z_limits)
    ax.set_xlim(mid_x - max_range/2, mid_x + max_range/2)
    ax.set_ylim(mid_y - max_range/2, mid_y + max_range/2)
    ax.set_zlim(mid_z - max_range/2, mid_z + max_range/2)

    for idx, vertex in enumerate(vertices):
        position = vertex[:3]
        orientation = vertex[3:]

        # Plot the position
        ax.scatter(position[0], position[1], position[2], color='blue')

        # Plot the vertex number
        ax.text(position[0], position[1], position[2], f'{idx}', color='black')

        # Plot the orientation using coordinate axes
        rotation = R.from_quat(orientation)

        # Define unit vectors in the local coordinate frame
        unit_x = rotation.apply([1, 0, 0])
        unit_y = rotation.apply([0, 1, 0])
        unit_z = rotation.apply([0, 0, 1])

        # Plot the coordinate axes
        ax.quiver(position[0], position[1], position[2], 
                  unit_x[0], unit_x[1], unit_x[2], color='r', length=scale)
        ax.quiver(position[0], position[1], position[2], 
                  unit_y[0], unit_y[1], unit_y[2], color='g', length=scale)
        ax.quiver(position[0], position[1], position[2], 
                  unit_z[0], unit_z[1], unit_z[2], color='b', length=scale)

    # Set labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Poses')
    ax.set_box_aspect([1, 1, 1])  # Equal aspect ratio

    plt.show()

def plot_edges(vertices, edges):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Collect all positions to determine axis limits
    positions = np.array([v[:3] for v in vertices])
    x_limits = [np.min(positions[:, 0]), np.max(positions[:, 0])]
    y_limits = [np.min(positions[:, 1]), np.max(positions[:, 1])]
    z_limits = [np.min(positions[:, 2]), np.max(positions[:, 2])]
    max_range = max(x_limits[1] - x_limits[0], y_limits[1] - y_limits[0], z_limits[1] - z_limits[0])

    # Set equal axes limits
    mid_x = np.mean(x_limits)
    mid_y = np.mean(y_limits)
    mid_z = np.mean(z_limits)
    ax.set_xlim(mid_x - max_range/2, mid_x + max_range/2)
    ax.set_ylim(mid_y - max_range/2, mid_y + max_range/2)
    ax.set_zlim(mid_z - max_range/2, mid_z + max_range/2)

    for edge in edges:
        i, j, _, _ = edge
        pos_i = np.array(vertices[i][:3])
        pos_j = np.array(vertices[j][:3])

        # Plot the edge as an arrow
        ax.quiver(pos_i[0], pos_i[1], pos_i[2], 
                  pos_j[0] - pos_i[0], pos_j[1] - pos_i[1], pos_j[2] - pos_i[2], 
                  color='b', arrow_length_ratio=0.1)

        # Annotate the edge with the vertex IDs
        #mid_point = (pos_i + pos_j) / 2
        #ax.text(mid_point[0], mid_point[1], mid_point[2], f'{i}-{j}', color='red')

    # Set labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Edges')
    ax.set_box_aspect([1, 1, 1])  # Equal aspect ratio

    plt.show()

def read_g2o(file_path):
    vertices = []
    edges = []
    sigma_T = None
    sigma_R = None

    with open(file_path, 'r') as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split()
        if parts[0] == 'VERTEX_SE3:QUAT':
            idx = int(parts[1])
            pose = list(map(float, parts[2:]))
            vertices.append((idx, pose))
        elif parts[0] == 'EDGE_SE3:QUAT':
            i = int(parts[1])
            j = int(parts[2])
            pose = list(map(float, parts[3:10]))
            upper_triangular_information = list(map(float, parts[10:]))
            information_matrix = np.zeros((6, 6))
            indices = np.triu_indices(6)
            information_matrix[indices] = upper_triangular_information
            information_matrix[(indices[1], indices[0])] = upper_triangular_information  # Symmetrize the matrix
            edges.append((i, j, pose, information_matrix))

            # Recover sigma_T and sigma_R from information matrix
            if sigma_T is None and sigma_R is None:
                covariance_matrix = np.linalg.inv(information_matrix)
                sigma_T = np.sqrt(np.mean(np.diag(covariance_matrix)[:3]))
                sigma_R = np.sqrt(np.mean(np.diag(covariance_matrix)[3:]))

    # Sort vertices by index
    vertices.sort(key=lambda x: x[0])
    vertices = [v[1] for v in vertices]

    Nvertices = len(vertices)
    Nedges = len(edges)

    return Nvertices, Nedges, vertices, edges, sigma_T, sigma_R

def main():
    # Default file path
    file_path = '../data/tinyGrid3D.g2o'

    # Check if a different file path is provided as a command-line argument
    if len(sys.argv) > 1:
        file_path = sys.argv[1]

    Nvertices, Nedges, vertices, edges, sigma_T, sigma_R = read_g2o(file_path)

    print("Nvertices:", Nvertices)
    print("Nedges:", Nedges)
    #print("Vertices:", vertices)
    #print("Edges:", edges)
    print("sigma_T:", sigma_T)
    print("sigma_R:", sigma_R)

    start_time = time.time()
    if (do_solve_sdp):
        Q, A_list, b_list = construct_sdp(Nvertices, edges, sigma_T, sigma_R, do_constraints=True)
    else:
        Q, A_list, b_list = construct_sdp(Nvertices, edges, sigma_T, sigma_R)
    end_time = time.time()
    run_time = (end_time - start_time) * 1000
    print(f"construct_sdp time: {run_time:.2f} milliseconds")

    Q_tild = retrieve_Q_tild_from_Q(Q)
    savemat('data.mat', {'Q':Q_tild, 'n':Nvertices})

    # Optionally, plot the matrix using matplotlib
    #plt.imshow(Q_tild, cmap='viridis')
    plt.imshow(Q_tild, cmap='jet')
    #plt.imshow(Q_tild, cmap='prism')
    plt.colorbar()
    plt.title(r'Matrix $\tilde{Q}$')
    plt.show()

    if (do_solve_sdp):
        start_time = time.time()
        r_hat = solve_sdp(Q, A_list, b_list)
        end_time = time.time()
        run_time = (end_time - start_time) * 1000
        print(f"solve_sdp time: {run_time:.2f} milliseconds")

        # Let's check
        R_hat = construct_R_from_r(r_hat, Nvertices)
        r_check = construct_r_from_R(R_hat)

        # trace(Q_tild @ R.T @ R) = vec(R) @ Q @ vec(R)
        # Q = kron(Q_tild, I3)

        print('SDPval: ', r_hat.T @ Q @ r_hat)
        print('SDPval check: ', np.trace(R_hat.T @ Q_tild @ R_hat))
        print('SDPval check: ', np.trace(Q_tild @ R_hat @ R_hat.T))

    if (do_solve_smaller_sdp):
        start_time = time.time()
        smaller_R_hat = solve_smaller_sdp(Q_tild)
        end_time = time.time()
        run_time = (end_time - start_time) * 1000
        print(f"solve_smaller_sdp time: {run_time:.2f} milliseconds")

    if (do_solve_manopt_SO3):
        R_manopt_SO3 = solve_problem_so3(Q_tild, Nvertices)
        result_manopt_SO3 = np.trace(Q_tild @ R_manopt_SO3 @ R_manopt_SO3.T)
        print("\n\033[92mManopt SO3 results:\033[0m")
        print('Result manopt SO3: ', result_manopt_SO3)

    if (do_solve_staircase):
        d = 3
        r0 = 2
        n = Nvertices
        dn = d * n
        R_staircase = solve_problem(Q_tild, d, r0, dn, n)
        result_staircase = np.trace(Q_tild @ R_staircase @ R_staircase.T)
        print("\n\033[92mStaircase results:\033[0m")
        print('Result staircase: ', result_staircase)

    #for vertex in vertices:
    #    print(vertex)
    #    translation = vertex[:3]
    #    quaternion = vertex[3:]
    #    r = R.from_quat(quaternion)
    #    Rotation = r.as_matrix()
    #    print(Rotation)

    plot_poses(vertices)
    plot_edges(vertices, edges)

if __name__ == "__main__":
    main()

