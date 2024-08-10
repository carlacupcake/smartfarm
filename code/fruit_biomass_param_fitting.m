function fruit_biomass_param_fitting()

% Apply concept of superposition (actually improper to apply here b/c we
% have a first order nonlinear ODE) to get initial guesses for the DE model
% parameters, tune the params later with a neural net (PINN? NODE?)

days_to_maturity = 10;
N = days_to_maturity*24 + 30*24;
dt = 1;
c0 = 0.01; % threshold already met
P0 = 0.01;
c = c0*ones(N, 1);
P = P0*ones(N, 1);

% For canopy biomass model
ce = 0.05;
R0 = 2045;
kappa = sin(pi/8);
rhostd = 1;
A = 15e-4;
kc = 1;
dc = 0.1;

% For fruit biomass model
ap = 35;
Tsp = 0.8;
dp = 0.002;

for n=1:N-1
    dcdt = ce * R0 * (1 - exp(-kappa/rhostd * 3 * c(n)/A * sqrt(pi/A) * tan(pi/4))) * A * c(n) * (1 - c(n)/kc)- dc*c(n);
    if c(n) > 0.1
        dPdt = ap * Tsp * dcdt * P(n) * (1 - P(n)/c(n)) - dp*P(n);
    else
        dPdt = ap * Tsp * dcdt * P(n) * (1 - P(n)/c(n));
    end
    c(n+1) = c(n) + dcdt * dt;
    P(n+1) = P(n) + dPdt * dt;
end

figure(1)
%plot(1:N, c)
hold on
plot(1:N, P)

end