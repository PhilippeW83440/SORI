import numpy as np
from scipy.spatial.transform import Rotation as R
import random

# Define constants for noise
sigma_T = 0.1  # Standard deviation for translational noise
sigma_R = 0.05  # Standard deviation for rotational noise (in radians)

# Parameters
Nvertices = 9
Nedges = 15 # At least 15 to get a well-posed problem
Nedges = 25 # At least 15 to get a well-posed problem

# Generate Nvertices with random poses
vertices = []
for _ in range(Nvertices):
    x, y, z = np.random.uniform(-10, 10, 3)  # Random translation in range [-10, 10]
    rot = R.random().as_quat()  # Random rotation as quaternion
    vertices.append([x, y, z, *rot])
vertices = np.array(vertices)

# Generate Nedges with unique random pairs of vertices
edges = set()
while len(edges) < Nedges:
    i, j = random.sample(range(Nvertices), 2)
    if (i, j) not in edges and (j, i) not in edges:
        edges.add((i, j))
edges = list(edges)

def add_noise(pose, sigma_T, sigma_R):
    noisy_translation = pose[:3] + np.random.normal(0, sigma_T, 3)
    true_rotation = R.from_quat(pose[3:])
    noise_rotation = R.from_rotvec(np.random.normal(0, sigma_R, 3))
    noisy_rotation = (noise_rotation * true_rotation).as_quat()
    return np.hstack((noisy_translation, noisy_rotation))

# Define relative poses and add noise
relative_poses = []
for i, j in edges:
    # Extract absolute poses
    abs_pose_i = vertices[i]
    abs_pose_j = vertices[j]

    # Calculate true relative translation
    translation_i = abs_pose_i[:3]
    translation_j = abs_pose_j[:3]
    relative_translation = translation_j - translation_i

    # Calculate true relative rotation
    rotation_i = R.from_quat(abs_pose_i[3:])
    rotation_j = R.from_quat(abs_pose_j[3:])
    relative_rotation = rotation_j * rotation_i.inv()

    true_relative_pose = np.hstack((relative_translation, relative_rotation.as_quat()))

    # Add noise to the true relative pose
    noisy_relative_pose = add_noise(true_relative_pose, sigma_T, sigma_R)

    relative_poses.append((i, j, noisy_relative_pose))

# Compute the information matrix
# Covariance matrix with translational and rotational noise
covariance_matrix = np.zeros((6, 6))
covariance_matrix[:3, :3] = np.eye(3) * sigma_T**2
covariance_matrix[3:, 3:] = np.eye(3) * sigma_R**2
information_matrix = np.linalg.inv(covariance_matrix)
upper_triangular_indices = np.triu_indices(6)
upper_triangular_information = information_matrix[upper_triangular_indices]

# Prepare output
output = []

# Add vertices to output
for idx, vertex in enumerate(vertices):
    output.append(f"VERTEX_SE3:QUAT {idx} {' '.join(f'{v:.6f}' for v in vertex)}")

# Add edges to output
information_matrix_str = ' '.join(f'{v:.6f}' for v in upper_triangular_information)
for edge in relative_poses:
    i, j, pose = edge
    output.append(f"EDGE_SE3:QUAT {i} {j} {' '.join(f'{v:.6f}' for v in pose)} {information_matrix_str}")

# Print the result
for line in output:
    print(line)

# Write the result to a file
with open("../data/random.g2o", "w") as f:
    for line in output:
        f.write(line + "\n")
