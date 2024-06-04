import numpy as np
from scipy.spatial.transform import Rotation as R
import random

N = 4

def generate_path():
    path = []
    visited = set()

    # Initial point
    current = (0, 0, N)
    path.append(current)
    visited.add(current)

    def is_valid(x, y, z):
        return 0 <= x <= N and 0 <= y <= N and 0 <= z <= N

    # Possible steps
    steps = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]

    while len(visited) < (N + 1) ** 3:  # 5*5*5 = 125 points in total
        for dx, dy, dz in steps:
            next_point = (current[0] + dx, current[1] + dy, current[2] + dz)
            if is_valid(*next_point) and next_point not in visited:
                current = next_point
                visited.add(current)
                path.append(current)
                break

    return path

# Generate the path
path = generate_path()

def generate_vertices(path):
    vertices = []
    vertex_map = {}
    for idx, (x, y, z) in enumerate(path):
        # Random rotation as quaternion
        rot = R.random().as_quat()
        vertices.append([x, y, z, *rot])
        vertex_map[(x, y, z)] = idx
    return vertices, vertex_map

vertices, vertex_map = generate_vertices(path)

def generate_edges(vertices, vertex_map, probability=0.3):
    edges = set()
    steps = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]
    N = len(vertices)

    # Ensure we have an edge between i and i+1 for odometry measurements
    for i in range(N - 1):
        edges.add((i, i + 1))

    for idx, (x, y, z, *_) in enumerate(vertices):
        for step in steps:
            if random.random() < probability:
                dx, dy, dz = step
                neighbor = (x + dx, y + dy, z + dz)
                if neighbor in vertex_map:
                    neighbor_idx = vertex_map[neighbor]
                    if (idx, neighbor_idx) not in edges:
                        edges.add((idx, neighbor_idx))

    #for idx, (x, y, z, *_) in enumerate(vertices):
    #    if random.random() < probability:
    #        for _ in range(10):
    #            dx, dy, dz = random.choice(steps)
    #            neighbor = (x + dx, y + dy, z + dz)
    #            if neighbor in vertex_map:
    #                neighbor_idx = vertex_map[neighbor]
    #                if (idx, neighbor_idx) not in edges and (neighbor_idx, idx) not in edges:
    #                    edges.add((idx, neighbor_idx))
    #                    #break

    return list(edges)


def add_noise(pose, sigma_T, sigma_R):
    noisy_translation = pose[:3] + np.random.normal(0, sigma_T, 3)
    true_rotation = R.from_quat(pose[3:])
    noise_rotation = R.from_rotvec(np.random.normal(0, sigma_R, 3))
    noisy_rotation = (noise_rotation * true_rotation).as_quat()
    return np.hstack((noisy_translation, noisy_rotation))

sigma_T = 0.1
sigma_R = 0.05

def generate_relative_poses(vertices, edges, sigma_T, sigma_R):
    relative_poses = []
    for i, j in edges:
        pose_i = np.array(vertices[i][:7])
        pose_j = np.array(vertices[j][:7])

        # Calculate true relative translation
        translation_i = pose_i[:3]
        translation_j = pose_j[:3]
        relative_translation = translation_j - translation_i

        # Calculate true relative rotation
        rotation_i = R.from_quat(pose_i[3:])
        rotation_j = R.from_quat(pose_j[3:])
        relative_rotation = rotation_j * rotation_i.inv()

        true_relative_pose = np.hstack((relative_translation, relative_rotation.as_quat()))

        # Add noise to the true relative pose
        noisy_relative_pose = add_noise(true_relative_pose, sigma_T, sigma_R)

        relative_poses.append((i, j, noisy_relative_pose))

    return relative_poses

edges = generate_edges(vertices, vertex_map)
relative_poses = generate_relative_poses(vertices, edges, sigma_T, sigma_R)

print("Vertices:")
for idx, vertex in enumerate(vertices):
    print(f"VERTEX_SE3:QUAT {idx} {' '.join(f'{v:.6f}' for v in vertex)}")

print("Edges:")
covariance_matrix = np.zeros((6, 6))
covariance_matrix[:3, :3] = np.eye(3) * sigma_T**2
covariance_matrix[3:, 3:] = np.eye(3) * sigma_R**2
information_matrix = np.linalg.inv(covariance_matrix)
upper_triangular_indices = np.triu_indices(6)
upper_triangular_information = information_matrix[upper_triangular_indices]
information_matrix_str = ' '.join(f'{v:.6f}' for v in upper_triangular_information)

for edge in relative_poses:
    i, j, pose = edge
    print(f"EDGE_SE3:QUAT {i} {j} {' '.join(f'{v:.6f}' for v in pose)} {information_matrix_str}")

# Write the result to a file
with open('../data/grid' + str(N) + '.g2o', "w") as f:
    for idx, vertex in enumerate(vertices):
        f.write(f"VERTEX_SE3:QUAT {idx} {' '.join(f'{v:.6f}' for v in vertex)}\n")
    for edge in relative_poses:
        i, j, pose = edge
        f.write(f"EDGE_SE3:QUAT {i} {j} {' '.join(f'{v:.6f}' for v in pose)} {information_matrix_str}\n")

