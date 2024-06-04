# Import packages.
import cvxpy as cp
import numpy as np

# Suppose X_opt is the solution obtained from the solve_sdp function
def retrieve_r_rank1(X_opt):
    # Perform SVD
    U, S, VT = np.linalg.svd(X_opt)

    # Take the first singular value and the corresponding singular vectors
    r = np.sqrt(S[0]) * U[:, 0]
    return r

# Function to retrieve r from X using all singular values
def retrieve_r_all_singular_values(X_opt):
    # Perform SVD
    U, S, VT = np.linalg.svd(X_opt)
    # Construct r by combining singular values and left singular vectors
    # Sum up contributions of all singular values
    r = np.dot(U, np.sqrt(S))
    return r

# Function to retrieve r from X using the top k singular values
def retrieve_r_top_k_singular_values(X_opt, k):
    # Perform SVD
    U, S, VT = np.linalg.svd(X_opt)

    # Use the top k singular values and vectors
    U_k = U[:, :k]
    S_k = S[:k]
    # Construct r by combining the top k singular values and vectors
    r = U_k @ np.sqrt(S_k)
    return r

def nearest_rotation(R_i):
    # Compute the SVD of the matrix
    U, _, Vt = np.linalg.svd(R_i)
    # Compute the nearest rotation matrix
    R_nearest = U @ Vt
    # Ensure it's a proper rotation matrix
    if np.linalg.det(R_nearest) < 0:
        U[:, -1] = -U[:, -1]
        R_nearest = U @ Vt
    return R_nearest


# Generate a random SDP.
def solve_sdp(Q, A_list, b_list):

    assert Q.shape[0] == Q.shape[1]

    n = Q.shape[0] # X of size
    p = len(b_list) # constraints

    assert n % 9 == 0

    m = int(n / 9)

    # Define and solve the CVXPY problem.
    # Create a symmetric matrix variable.
    X = cp.Variable((n,n), symmetric=True)
    # The operator >> denotes matrix inequality.
    constraints = [X >> 0]
    constraints += [
        cp.trace(A_list[i] @ X) == b_list[i] for i in range(p)
    ]
    prob = cp.Problem(cp.Minimize(cp.trace(Q @ X)),
                      constraints)
    prob.solve(solver=cp.SCS, verbose=True)

    # Print result.
    print("The optimal value is", prob.value)

    X_opt = X.value
    evals, _ = np.linalg.eig(X_opt)
    print('evals: ', evals)

    #r_opt = retrieve_r_rank1(X_opt)
    #r_opt = retrieve_r_all_singular_values(X_opt)
    r_opt = retrieve_r_top_k_singular_values(X_opt, 9)

    reflector = np.diag([1] * (3 - 1) + [-1])

    r_hat = np.zeros_like(r_opt)
    idx_start = 0
    for i in range(m):
        R_opt = r_opt[idx_start:idx_start+9]
        R_opt = R_opt.reshape(3, 3, order='F') # colums by columns

        print('determinant: ', np.linalg.det(R_opt))
        if np.linalg.det(R_opt) < 0:
            R_opt = reflector * R_opt

        R_hat = nearest_rotation(R_opt)
        r_hat[idx_start:idx_start+9] = R_hat.flatten(order='F')

        idx_start += 9

    X_hat = np.outer(r_hat, r_hat)
    lower_bound = prob.value
    upper_bound = np.trace(Q @ X_hat)

    print("\n\033[92mSDP results:\033[0m")
    print('Value of SDP solution is', lower_bound)
    print('Value of rounded pose estimate is', upper_bound)
    print('Suboptimality bound of recovered solution is: {} \n'.format(upper_bound - lower_bound))

    return r_hat
