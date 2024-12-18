% Number of Data points in each time step
N_point = 22500;

% Number of time steps
N_t = 77;

% Define U_total
U_total = zeros(N_point * N_t, 1);

for i = 1:N_t
    U = density(:, :, i);
    U(isnan(U)) = 0;  % Replace NaN values with 0
    U = reshape(U, N_point, 1);
    U_total((i - 1) * N_point + 1:i * N_point) = U;
end

% Save matrix
U_1 = [U_total];
save('U_total_1.mat', 'U_1');

