function Y_star = solve_problem(Q, n)

    d = 3
    dn = d * n
    r0 = 6
    rmax = 10

    % Initialize the random seed for reproducibility
    rng(1);

    % Compute the Jacobi preconditioner (inverse of the diagonal elements of Q)
    P = 1 ./ diag(Q);

    for r = r0:rmax
        disp(['r: ', num2str(r)]);
        
        % Define the manifold using Manopt
        manifold = stiefelstackedfactory(n, d, r);

        % Define the cost function
        cost = @(X) cost_function(X, Q);

        % Define the gradient of the cost function
        egrad = @(X) egrad_function(X, Q, P);

        % Define the Hessian of the cost function
        ehess = @(X, U) ehess_function(X, U, Q, P);

        % Set up the problem structure
        problem.M = manifold;
        problem.cost = cost;
        problem.egrad = egrad;
        problem.ehess = ehess;

        % Solve the problem using the trust-regions solver
        options.tolgradnorm = 1e-2;
        options.rel_func_tol = 1e-5;
        options.miniter = 1;
        options.maxiter = 300;
        options.maxinner = 500;
        [Y_star, ~] = trustregions(problem, [], options);

        % Check rank condition and return if satisfied
        Y_star = mat2cell(Y_star, d*ones(n, 1), r);
        if any(cellfun(@(y) rank(y) < r, Y_star))
            Y_star = cell2mat(Y_star);
            return;
        else
            if r < dn
                Y_star = cellfun(@(y) padarray(y, [0, d - size(y, 2)], 0, 'post'), Y_star, 'UniformOutput', false);
            end
        end
    end

    Y_star = cell2mat(Y_star);
end

function f = cost_function(X, Q)
    f = trace(X' * Q * X);
end

function g = egrad_function(X, Q, P)
    g = 2 * Q * X;
    %g = bsxfun(@times, P, g); % Apply Jacobi preconditioner
end

function h = ehess_function(X, U, Q, P)
    h = 2 * Q * U;
    %h = bsxfun(@times, P, h); % Apply Jacobi preconditioner
end

