# Import packages.
import cvxpy as cp
import numpy as np

def recover_R_from_X(X, d=3):
    # Perform SVD on X
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    # Extract the top d singular values and corresponding singular vectors
    S_d = np.diag(np.sqrt(S[:d]))
    V_d = Vt[:d, :]
    # Compute R as the product of the square root of the singular values and the corresponding right singular vectors
    R = np.dot(S_d, V_d)
    return R


def nearest_rotation(R_i):
    # Compute the SVD of the matrix
    U, _, Vt = np.linalg.svd(R_i)
    # Compute the nearest rotation matrix
    R_nearest = U @ Vt
    # Ensure it's a proper rotation matrix
    if np.linalg.det(R_nearest) < 0:
        U[:, -1] *= -1
        R_nearest = U @ Vt
    return R_nearest


# Q is if shape nd x nd
# With n the number of rotations
# and d=3
def solve_smaller_sdp(Q):

    assert Q.shape[0] == Q.shape[1]

    nd = Q.shape[0] # Q of size nd x nd
    assert nd % 3 == 0
    d = 3

    n = int(nd / d)

    # Define the variable Z
    X = cp.Variable((d * n, d * n), symmetric=True)
    constraints = [X >> 0]  # Z should be positive semidefinite

    # Add block diagonal constraints for Id blocks
    Id = np.eye(d)
    for i in range(n):
        constraints.append(X[i*d:(i+1)*d, i*d:(i+1)*d] == Id)

    # Define the objective function
    objective = cp.Minimize(cp.trace(Q @ X))

    # Define the problem
    problem = cp.Problem(objective, constraints)

    # Solve the problem using SCS solver
    #problem.solve(solver=cp.SCS, verbose=True, eps_abs=1e-3, eps_rel=1e-3, eps_infeas=1e-5, max_iters=int(1e4), time_limit_secs=180)
    problem.solve(solver=cp.SCS, verbose=True, eps_abs=1e-4,  eps_rel=1e-4, max_iters=int(1e4), time_limit_secs=600)

    # Print result.
    print("The optimal value is", problem.value)

    X_opt = X.value

    evals, _ = np.linalg.eig(X_opt)
    print('evals: ', evals)

    X_opt = 0.5 * (X_opt + X_opt.T)

    R_opt = recover_R_from_X(X_opt) # size 3nx3: the n rotation matrices are stacked vertically

    determinants = np.zeros(n)
    for k in range(n):
        submatrix = R_opt[:, d * k : d * (k + 1)]
        determinants[k] = np.linalg.det(submatrix)
    ng0 = np.sum(determinants > 0)
    reflector = np.diag([1] * (d - 1) + [-1])
    if ng0 == 0:
        # This solution converged to a reflection of the correct solution
        R_opt = np.dot(reflector, R_opt)
        determinants = -determinants
    elif ng0 < n:
        # If more than half of the determinants have negative sign, reverse them
        if ng0 < n / 2:
            determinants = -determinants
            R_opt = np.dot(reflector, R_opt)

    R_hat = np.zeros_like(R_opt)
    for i in range(n):
        Ri = R_opt[:, i*3:(i+1)*3]
        R_hat[:, i*3:(i+1)*3] = nearest_rotation(Ri)

    X_hat = R_hat.T @ R_hat
    lower_bound = problem.value
    upper_bound = np.trace(Q @ X_hat)

    print("\n\033[92mSmaller SDP results:\033[0m")
    print('Value of SDP solution is', lower_bound)
    print('Value of rounded pose estimate is', upper_bound)
    print('Suboptimality bound of recovered solution is: {} \n'.format(upper_bound - lower_bound))

    return R_hat
