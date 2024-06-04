import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.sparse import rand, csr_matrix

# f_ml = min sum_ij (norm(tj - ti - Ri @ t_ij)**2 + norm(Rj - Ri @ R_ij)**2) 

# variables   : ti in R3 and Ri in SO(3)
# measurements: t_ij, R_ij corresponding to relative poses 
# edges: ij



# get Extraction matrix for (tj - ti - Tij @ ri) from [t1,...,tm,r1,...,rm]

def getBij(i, j, Tij, m):

    assert Tij.shape[0] == 3
    assert Tij.shape[1] == 9

    # Initialize B_ij as a 3 x 12m zero matrix
    Bij = csr_matrix((3, 12 * m))

    # Positions of t_i, t_j, and r_i
    t_i_start = 3 * i
    t_i_end = 3 * (i + 1)
    t_j_start = 3 * j
    t_j_end = 3 * (j + 1)
    r_i_start = 3 * m + 9 * i
    r_i_end = 3 * m + 9 * (i + 1)

    # Setting the entries for t_j - t_i
    Bij[:, t_j_start:t_j_end] = np.eye(3)  # +I for t_j
    Bij[:, t_i_start:t_i_end] = -np.eye(3)  # -I for t_i

    # Setting the entries for -T_ij * r_i
    Bij[:, r_i_start:r_i_end] = -Tij

    return Bij


# get Extraction matrix for (rj - Qij @ ri) from [t1,...,tm,r1,...,rm]

def getCij(i, j, Qij, m):

    assert Qij.shape[0] == 9
    assert Qij.shape[1] == 9

    # Initialize C_ij as a 9 x 12m zero matrix
    Cij = csr_matrix((9, 12 * m))

    # Positions of r_i and r_j
    r_i_start = 3 * m + 9 * i
    r_i_end = 3 * m + 9 * (i + 1)
    r_j_start = 3 * m + 9 * j
    r_j_end = 3 * m + 9 * (j + 1)

    # Setting the entries for r_j - Q_ij * r_i
    Cij[:, r_j_start:r_j_end] = np.eye(9)  # +I for r_j
    Cij[:, r_i_start:r_i_end] = -Qij  # -Qij for r_i

    return Cij

# For every rotation matrix:
#   c_j^T c_j = 1 constraint
#   c_j^T c_k = 0 constraint

def get_rotation_constraints(m):
    A_list = []
    b_list = []

    for i in range(m):
        # Indices for the rotation matrix r_i in the vector r
        r_i_start = 9 * i
        r_i_end = 9 * (i + 1)

        for j in range(3):
            for k in range(j, 3):
                A_l = np.zeros((9 * m, 9 * m))
                if j == k:
                    # c_j^T c_j = 1 constraint
                    idx = r_i_start + 3 * j
                    A_l[idx:idx + 3, idx:idx + 3] = np.eye(3)
                    b_l = 1.0
                else:
                    # c_j^T c_k = 0 constraint
                    idx_j = r_i_start + 3 * j
                    idx_k = r_i_start + 3 * k
                    A_l[idx_j:idx_j + 3, idx_k:idx_k + 3] = np.eye(3)
                    A_l[idx_k:idx_k + 3, idx_j:idx_j + 3] = np.eye(3)
                    b_l = 0.0

                A_list.append(A_l)
                b_list.append(b_l)

    return A_list, b_list


def construct_sdp(Nvertices, edges, sigma_T, sigma_R, do_constraints=False):

    # Project report notations
    m = Nvertices
    w_t = 1./sigma_T
    w_R = 1./sigma_R

    w_R *= np.sqrt(1/2)

    A = csr_matrix((12 * m, 12 * m))

    A_list = None
    b_list = None

    I = np.eye(3)

    num_edge = 0
    for edge in edges:
        num_edge += 1
        print('num_edge: ', num_edge)

        if (num_edge % 100) == 0:
            print('NUM edge: ', num_edge)
        i, j, pose, _ = edge

        assert i < m
        assert j < m

        t_ij = np.array(pose[:3]) # t_ij measurement

        quaternion = pose[3:]
        quaternion = quaternion / np.linalg.norm(quaternion)
        r = R.from_quat(quaternion)
        R_ij = r.as_matrix() # R_ij measurement

        # vec(AXB) = (B.T x A) @ vec(X)

        # vec(I Ri t_ij) = kron(t_ij.T, I) @ vec(Ri)
        Tij = np.kron(t_ij.T, I) # (3, 9)
        # vec(I Ri R_ij) = kron(R_ij.T, I) @ vec(Ri)
        Qij = np.kron(R_ij.T, I) # (9, 9)

        Bij = getBij(i, j, Tij, m)
        Cij = getCij(i, j, Qij, m)

        Bij *= w_t
        Cij *= w_R

        A += (Bij.T @ Bij)
        A += (Cij.T @ Cij)

    # ----------------------------------
    # Now we have the problem in QP form
    # ----------------------------------
    # f_QP = min x.T @ A @ x
    # with x = [t1,...,tm,r1,...,rm].T of size 12m unknow
    # A of size (12m, 12m) given

    # constraints: ti in R3, Ri in SO(3)

    A = A.toarray()

    Att = A[:3*m, :3*m]
    Atr = A[:3*m,  3*m:]
    Art = A[ 3*m:, :3*m]
    Arr = A[ 3*m:,  3*m:]

    # Note that if this matrix is not invertible, the problem is not well posed
    # For any vector u in the nullspace of Att Att @ u = 0
    # If t* is solution then t*+u is solution as well 
    Att += 1e-8 * np.eye(3*m)
    Att_inv = np.linalg.inv(Att)

    # ------------------------------------------
    # Now we get the problem in reduced QP form
    # ------------------------------------------
    # f_QP = min r.T @ Q @ r
    # with r = [r1,...,rm].T of size 9m unknow
    # Q of size (9m, 9m) given

    # constraints: Ri in SO(3)

    # we can then recover the translations part
    # t* = - Att_inv @ Atr @ r

    Q = Arr - Art @ Att_inv @ Atr

    # Constraints:
    # r.T @ Al @ r = bl 

    # for l in [1,6m]
    # We have 6 constraints (so far) per rotation matrix:
    #   cj.T cj = 1 for j=1,2,3
    #   cj.T ck = 0 for j != k
    # r  is of size  (9m,)
    # Al are of size (9m, 9m)
    # bl is a scalar

    # A_list: 6m np.array of size (9m, 9m)
    #         A matrices are full of zeros except for 1 diagonal block of size (9, 9)
    # b_list: 6m scalars
    if (do_constraints):
        A_list, b_list = get_rotation_constraints(m) # 6m matrices of size (9m, 9m)
        b = np.array(b_list) # (6m,)

    # Now the problem is in the following form

    # -------------------------------------------
    # min r.T @ Q @ r
    # st  r.T @ Ai @ r = bi  for i in [1,6m]
    # -------------------------------------------

    # r  is of size  (9m,): m rotation vectors

    #evals, _ = np.linalg.eig(Q)
    #print('Q evals: ', evals)

    return Q, A_list, b_list
