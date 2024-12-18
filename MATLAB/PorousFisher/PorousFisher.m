% Define constants
L_left = 0;
L_right = 4380;
nx = 150;
ny = nx;
dx = (L_right - L_left) / (nx - 1);
dy = dx;
h = dx;

% Define Mesh
x = linspace(L_left, L_right, nx);
y = linspace(L_left, L_right, ny);
[X, Y] = meshgrid(x, y);
M = length(x);

% Time Steps
dt = 1/3;
N_t = 77;       % Total time steps
N_point = nx * ny;    % Total grid points
%% Parameteres 
% free of PINN
D_0_pinn = 1147;
r_pinn = 0.3952;
K_pinn = 2970;
eta_pinn = 0.0128;
% free of MLE
D_0_mle = 1159;
r_mle = 0.3824;
K_mle = 2882;
eta_mle = 0.1641;

% fixed
alpha = 1;
beta = 1;
gamma = 1;

% Initialize the concentration field
U_pinn = zeros(ny,nx);
U_mle = U_pinn;
%circle
U_pinn = sqrt((X-L_right/2).^2 + (Y-L_right/2).^2) < 0.25*L_right;
U_mle = U_pinn;
U_pinn = U_pinn*K_pinn;
U_mle = U_mle*K_mle;

% Plot at initial condition
figure
pcolor(X, Y, U_pinn); shading interp;
hold on
xlabel('x values');
ylabel('y values');

% Flatten grid for vectorized operations
U_pinn = U_pinn(:);
U_mle = U_mle(:);
% Define the coefficients for PINN
D_pinn = @(U) D_0_pinn*(U/K_pinn).^eta_pinn;
f_pinn = @(U) r_pinn*U.^alpha.*(abs(1-(U/K_pinn)).^gamma).^beta.*sign(1-(U/K_pinn));

% Define the coefficients for MLE
D_mle = @(U) D_0_mle*(U/K_mle).^eta_mle;
f_mle = @(U) r_mle*U.^alpha.*(abs(1-(U/K_mle)).^gamma).^beta.*sign(1-(U/K_mle));

% Construct the finite volume matrices
I = speye(nx * ny);
e = ones(nx, 1);
Lx = spdiags([e -2*e e], [-1 0 1], nx, nx);
Ly = spdiags([e -2*e e], [-1 0 1], ny, ny);

% Apply zero Neumann boundary conditions
Lx(1, 2) = 2; Lx(nx, nx-1) = 2; % Neumann BC
Ly(1, 2) = 2; Ly(ny, ny-1) = 2; % Neumann BC

Dx = kron(I(1:ny, 1:ny), Lx) / dx^2;
Dy = kron(Ly, I(1:nx, 1:nx)) / dy^2;
Lap = Dx + Dy;

% Time stepping loop
U_pinn_total = zeros(N_point*N_t,1);
for n = 1:N_t
    D_U_pinn = D_pinn(U_pinn);
    A_left = I - dt / 2 * spdiags(D_U_pinn, 0, nx * ny, nx * ny) * Lap;
    A_right = I + dt / 2 * spdiags(D_U_pinn, 0, nx * ny, nx * ny) * Lap;

    % Compute the source term at the current time step
    f_U_pinn = f_pinn(U_pinn);
    
    % Compute the right-hand side
    b_pinn = A_right*U_pinn + dt*f_U_pinn;
    
    % Solve the linear system for the next time step
    U_pinn = A_left \ b_pinn;
    U_pinn_total((i - 1) * N_point + 1:i * N_point) = U_pinn;
end

for n = 1:N_t
    D_U_mle = D_mle(U_mle);
    A_left = I - dt / 2 * spdiags(D_U_mle, 0, nx * ny, nx * ny) * Lap;
    A_right = I + dt / 2 * spdiags(D_U_mle, 0, nx * ny, nx * ny) * Lap;

    % Compute the source term at the current time step
    f_U_mle = f_mle(U_mle);
    
    % Compute the right-hand side
    b_mle = A_right*U_mle + dt*f_U_mle;
    
    % Solve the linear system for the next time step
    U_mle = A_left \ b_mle;
end
% Plot at final time
figure
U_pinn = reshape(U_pinn, nx, ny);
pcolor(X, Y, U_pinn); shading interp;
xlabel('x values');
ylabel('y values');

% Compute error
U_pinn = reshape(U_pinn, nx*ny, 1);
U_exact = density(1:150,1:150,77);
U_exact(isnan(U_exact)) = 0;
U_exact = reshape(U_exact, 150*150, 1);
Error_pinn = sqrt(sum(((U_pinn - U_exact)).^2)/(nx*ny));
Error_mle = sqrt(sum(((U_mle - U_exact)).^2)/(nx*ny));

% Display results
disp(['Error_pinn: ', num2str(Error_pinn)]);
disp(['Error_mle: ', num2str(Error_mle)]);

% Create vector of experimental data at all time steps
U_total = zeros(N_point * N_t, 1);
for i = 1:N_t
    U = density(:, :, i);
    U(isnan(U)) = 0;  % Replace NaN values with 0
    U = reshape(U, N_point, 1);
    U_total((i - 1) * N_point + 1:i * N_point) = U;
end

% Calculate AIC & BIC
n = N_point*N_t;        % Number of data points
k = 945; % Number of parameters in the Neural Network
residuals = U_total - U_pinn_total; % Model residuals

% Compute log-likelihood assuming Gaussian errors
sigma2 = var(residuals);       % Variance of residuals
logL = -0.5 * n * log(2 * pi * sigma2) - 0.5 * sum(residuals.^2) / sigma2;

% Formula
AIC = 2 * k - 2 * logL;
BIC = k * log(n) - 2 * logL;

% Display results
disp(['AIC: ', num2str(AIC)]);
disp(['BIC: ', num2str(BIC)]);
