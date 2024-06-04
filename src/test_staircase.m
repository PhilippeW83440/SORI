% Test the staircase function
Q = randn(27, 27);
Q = Q' * Q; % Make Q positive semidefinite
n = 9;

display(n)

data = load('data.mat');
Q = data.Q;
n = double(data.n)

Y_star = staircase(Q, n);

disp('Y_star:');
%disp(Y_star);
disp('Size of Y_star:');
disp(size(Y_star));

