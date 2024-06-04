import numpy as np
import autograd.numpy as anp

import pymanopt
from pymanopt.manifolds import Product, Stiefel
from pymanopt.manifolds import SpecialOrthogonalGroup
from pymanopt.manifolds import Oblique
from pymanopt import Problem
from pymanopt.optimizers import TrustRegions

np.random.seed(1)


def solve_problem(Q, d, r0, dn, n):
    dn = min(dn, d)  # Ensure dn does not exceed d

    #for r in range(r0, dn + 1):
    #for r in range(2, d+1):
    for r in range(3, 4):
        print('r: ', r)
        manifold = Product([Stiefel(d, r) for _ in range(n)])
        #manifold = Product([Oblique(d, r) for _ in range(n)])
        #manifold = Product([SpecialOrthogonalGroup(d) for _ in range(n)])

        # Define the cost function using the appropriate decorator
        @pymanopt.function.numpy(manifold)
        def cost(*X):
            X_stacked = np.vstack(X)
            return anp.trace(anp.dot(X_stacked.T, anp.dot(Q, X_stacked)))

        # Define the gradient of the cost function using the appropriate decorator
        @pymanopt.function.numpy(manifold)
        def egrad(*X):
            X_stacked = np.vstack(X)
            grad_stacked = 2 * anp.dot(Q, X_stacked)
            return [grad_stacked[i * d:(i + 1) * d] for i in range(n)]

        # Define the Hessian of the cost function using the appropriate decorator
        @pymanopt.function.autograd(manifold)
        def ehess(*X_and_U):
            n = len(X_and_U) // 2
            X = X_and_U[:n]
            U = X_and_U[n:]
            U_stacked = anp.vstack(U)
            hess_stacked = 2 * anp.dot(Q, U_stacked)
            return [hess_stacked[i * d:(i + 1) * d] for i in range(n)]

        problem = Problem(manifold=manifold, cost=cost, euclidean_gradient=egrad, euclidean_hessian=ehess)
        solver = TrustRegions(min_gradient_norm=1e-4)
        Y_star = solver.run(problem).point

        if any(np.linalg.matrix_rank(y) < r for y in Y_star):
            return np.vstack(Y_star)
        else:
            if r < dn:
                Y_star = [np.pad(y, ((0, 0), (0, d))) for y in Y_star]

    return np.vstack(Y_star)

