% Parameters
Lx = 4380; Ly = 4380; % Domain size
L_right = 4380;
nx = 150; ny = 150;   % Grid points
dx = Lx / nx; dy = Ly / ny;
dt = 1 / 3;           % Time step
T_steps = 77;         % Total time steps
N_point = nx * ny;    % Total grid points

% Initial value of free parameters
D0 = 1300; r = 0.3; K = 2600; gamma = 1.5;
% Value of fixed parameters
alpha = 1; beta = 1; eta = 0;
U_obs = zeros(N_point * T_steps, 1);
for i = 1:T_steps
    U_data = density(1:150, 1:150, i);
    U_data(isnan(U_data)) = 0;
    U_obs(N_point * (i - 1) + 1:N_point * i) = U_data;
end

% Scaling factors
scale_D0 = 1000;
scale_r = 0.1;
scale_K = 1000;
scale_gamma = 1;

% Scale initial guesses
initial_guess_scaled = [D0 / scale_D0, r / scale_r, K / scale_K, gamma / scale_gamma];
lb_scaled = [100 / scale_D0, 0.01 / scale_r, 500 / scale_K, 0 / scale_gamma];
ub_scaled = [10000 / scale_D0, 1 / scale_r, 5000 / scale_K, 9 / scale_gamma];

% Set optimization options
options = optimoptions('fmincon', ...
    'Display', 'iter', ...
    'Algorithm', 'interior-point', ...
    'MaxIterations', 10, ...  % Limit to 10 iterations
    'OptimalityTolerance', 1e-6, ... % Fine-tune the tolerance
    'StepTolerance', 1e-6);    % Fine-tune step size tolerance

% Optimize scaled parameters
optimal_params_scaled = fmincon(@(params_scaled) cost_function_scaled(params_scaled, U_obs, ...
    nx, ny, L_right, T_steps, scale_D0, scale_r, scale_K, scale_gamma, dx, dy, dt), ...
    initial_guess_scaled, [], [], [], [], lb_scaled, ub_scaled, [], options);

% Unscale the optimized parameters
optimal_params = [optimal_params_scaled(1) * scale_D0, ...
                  optimal_params_scaled(2) * scale_r, ...
                  optimal_params_scaled(3) * scale_K, ...
                  optimal_params_scaled(4) * scale_gamma];

% Display results
disp('Estimated Parameters:');
disp(['D0: ', num2str(optimal_params(1))]);
disp(['r: ', num2str(optimal_params(2))]);
disp(['K: ', num2str(optimal_params(3))]);
disp(['gamma: ', num2str(optimal_params(4))]);

% Function to solve the PDE
function U_total = solve_pde(D0, r, K, eta, alpha, gamma, beta, nx, ny, L_right, dx, dy, dt, T_steps)
    % Initialization
    U_total = zeros(nx * ny * T_steps, 1);
    U_mle = zeros(ny, nx);

    % Circle initialization
    [X, Y] = meshgrid(linspace(0, L_right, nx), linspace(0, L_right, ny));
    U_mle = sqrt((X - L_right / 2).^2 + (Y - L_right / 2).^2) < 0.25 * L_right;
    U_mle = U_mle * K;

    % Flatten grid
    U_mle = U_mle(:);

    % Define non-linear terms
    D_mle = @(U) D0 * (U / K).^eta;
    f_mle = @(U) r * U.^alpha .* (abs(1 - (U / K)).^gamma).^beta .* sign(1 - (U / K));

    % Sparse matrix setup
    I = speye(nx * ny);
    e = ones(nx, 1);
    Lx = spdiags([e -2 * e e], [-1 0 1], nx, nx);
    Ly = spdiags([e -2 * e e], [-1 0 1], ny, ny);

    % Neumann boundary conditions
    Lx(1, 2) = 2; Lx(nx, nx - 1) = 2;
    Ly(1, 2) = 2; Ly(ny, ny - 1) = 2;

    Dx = kron(speye(ny), Lx) / dx^2;
    Dy = kron(Ly, speye(nx)) / dy^2;
    Lap = Dx + Dy;

    % Time stepping
    for n = 1:T_steps
        D_U_mle = D_mle(U_mle);
        A_left = I - dt / 2 * spdiags(D_U_mle, 0, nx * ny, nx * ny) * Lap;
        A_right = I + dt / 2 * spdiags(D_U_mle, 0, nx * ny, nx * ny) * Lap;

        % Source term
        f_U_mle = f_mle(U_mle);

        % Right-hand side
        b_mle = A_right * U_mle + dt * f_U_mle;

        % Solve linear system
        U_mle = A_left \ b_mle;

        % Save to U_total
        U_total((n - 1) * nx * ny + 1:n * nx * ny) = U_mle;
    end
end

% Cost function for optimization
function error = cost_function_scaled(params_scaled, U_obs, nx, ny, L_right, T_steps, ...
                                       scale_D0, scale_r, scale_K, scale_gamma, dx, dy, dt)
    % Unscale the parameters
    D0 = params_scaled(1) * scale_D0;
    r = params_scaled(2) * scale_r;
    K = params_scaled(3) * scale_K;
    gamma = params_scaled(4) * scale_gamma;

    eta = 0; alpha = 1; beta = 1;

    % Numerical solution of PDE
    U_sim = solve_pde(D0, r, K, eta, alpha, gamma, beta, nx, ny, L_right, dx, dy, dt, T_steps);

    % Compute error
    error = sqrt((sum(((U_obs - U_sim)).^2)));
end
