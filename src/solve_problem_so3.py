import numpy as np
import autograd.numpy as anp

import pymanopt
from pymanopt import Problem
from pymanopt.manifolds import Product
from pymanopt.manifolds import SpecialOrthogonalGroup
from pymanopt.optimizers import ConjugateGradient
from pymanopt.optimizers import TrustRegions
from pymanopt.optimizers import SteepestDescent

def solve_problem_so3(Q, m):
    d, k = 3, 3  # Each block is 3x3

    # Define the manifold as a product of m Stiefel manifolds
    # manifold = Product([Stiefel(d, k) for _ in range(m)])
    manifold = Product([SpecialOrthogonalGroup(d) for _ in range(m)])

    # Define the cost function using the appropriate decorator
    @pymanopt.function.numpy(manifold)
    def cost(*X):
        X_stacked = np.vstack(X)
        return anp.trace(anp.dot(X_stacked.T, anp.dot(Q, X_stacked)))

    # Define the gradient of the cost function using the appropriate decorator
    @pymanopt.function.numpy(manifold)
    def euclidean_gradient(*X):
        X_stacked = np.vstack(X)
        grad_stacked = 2 * anp.dot(Q, X_stacked)
        return [grad_stacked[i * d:(i + 1) * d] for i in range(m)]

    # Define the Hessian of the cost function using the appropriate decorator
    @pymanopt.function.autograd(manifold)
    def euclidean_hessian(*X_and_U):
        n = len(X_and_U) // 2
        X = X_and_U[:n]
        U = X_and_U[n:]
        U_stacked = anp.vstack(U)
        hess_stacked = 2 * anp.dot(Q, U_stacked)
        return [hess_stacked[i * d:(i + 1) * d] for i in range(m)]

    # Create the problem with cost function, gradient, and Hessian
    problem = Problem(manifold=manifold, cost=cost, euclidean_gradient=euclidean_gradient, euclidean_hessian=euclidean_hessian)

    # Instantiate a solver
    solver = ConjugateGradient(min_gradient_norm=1e-4, min_step_size=1e-4, max_iterations=3000)
    #solver = TrustRegions(min_gradient_norm=1e-4, max_iterations=1e3)
    #solver = SteepestDescent()

    # Solve the problem
    result = solver.run(problem)

    # Retrieve the optimal solution
    optimal_blocks = result.point
    optimal_X = np.vstack(optimal_blocks)

    return optimal_X
