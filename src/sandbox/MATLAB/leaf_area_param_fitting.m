function leaf_area_param_fitting()

% Apply concept of superposition (actually improper to apply here b/c we
% have a first order nonlinear ODE) to get initial guesses for the DE model
% parameters, tune the params later with a neural net (PINN? NODE?)

days_to_maturity = 30;
N = days_to_maturity*24 + 30*24;
dt = 1;
A0 = 0.01;
A = A0*ones(N, 1);

kL = 15e-4; % leaf carrying capacity m2
Tc = 1500;
Teff = 23;
Topt = 25;
bLT = Tc/Teff;
cLT = 20;
LT = 5e-5; % goal is 15 cm2 per 30 days or 5e-5 m2 per day
aLT = 15;
dLT = 0.009;
TSL = 1 - abs(1-Teff/Topt);
Wc = 200;
Weff = 5.375;
bLW = Wc/Weff;
cLW = 20;
Lw = 1/10;
aLw = 0.001;
dLW = 0.003;
WSL = 1;
kLF = 0.7;
Lf = 1/10;
aLf = 0.008;

Weff = 5.375 * ones(N, 1); 
Wc = zeros(N, 1);  
for n=24:24:N-24
    Wc(n+1:n+24) = Wc(n-23:n) + Weff(n);
end

Feff = zeros(N, 1);
lastFeff = 0;
m = 24*30;
for n=1:m:N-m
    Feff(n) = 0.2 + lastFeff;
    lastFeff = Feff(n);
end

carrying_not_reached = true;
for n=1:N-1
    if A(n) < kL && carrying_not_reached
        dAdt = LT * aLT * Teff * exp(-((Tc - bLT*Teff)/(cLT*Teff))^2)...
               + Lw * aLw * Weff(n) * exp(-((Wc(n) - bLW*Weff(n))/(cLW*Weff(n)))^2)...
               + Lf * aLf * Feff(n) * (1 - Feff(n)/kLF);
    elseif A(n) >= kL
        dAdt = LT * aLT * Teff * exp(-((Tc - bLT*Teff)/(cLT*Teff))^2)...
               + Lw * aLw * Weff(n) * exp(-((Wc(n) - bLW*Weff(n))/(cLW*Weff(n)))^2)...
               + Lf * aLf * Feff(n) * (1 - Feff(n)/kLF)...
               - dLT * TSL * A(n) - dLW * WSL * A(n);
        carrying_not_reached = false;
    else
        dAdt = LT * aLT * Teff * exp(-((Tc - bLT*Teff)/(cLT*Teff))^2)...
               + Lw * aLw * Weff(n) * exp(-((Wc(n) - bLW*Weff(n))/(cLW*Weff(n)))^2)...
               + Lf * aLf * Feff(n) * (1 - Feff(n)/kLF)...
               - dLT * TSL * A(n) - dLW * WSL * A(n);
    end
    A(n+1) = A(n) + dAdt * dt;
end

figure(1)
plot(1:N, A)

% figure(1)
% hold on
% syms A(t);
% A0 = 15e-4; % only use to simulate after carrying capacity reached
% ode = diff(A, t) == LT * aLT * Teff * exp(-((Tc - bLT*Teff)/(cLT*Teff))^2)...
%                     + Lw * aLw * Weff * exp(-((Wc - bLW*Weff)/(cLW*Weff))^2)...
%                     + Lf * aLf * Feff * (1 - Feff/kLF)...
%                     - dLT * TSL * A(t) - dLW * WSL * A(t);
% cond = A(0) == A0;
% ySol(t) = dsolve(ode, cond);
% fplot(ySol(t), [0, N])
% legend('Location','northwest')

end