function plant_height_param_fitting()

% Apply concept of superposition (actually improper to apply here b/c we
% have a first order nonlinear ODE) to get initial guesses for the DE model
% parameters, tune the params later with a neural net (PINN? NODE?)

days_to_maturity = 62;
N = days_to_maturity*24 + 30*24;
dt = 1;
h0 = 0.01;
h = h0*ones(N, 1);
ah = 12/(days_to_maturity*24); % the higher the faster
kh = 1; % ideal plant height
dh = 0.05; % only have it kick in after carrying capacity is reached
Hw = 1/5 * kh /10; % for water contributes like 1/5 of height
Wc = 500;
Weff = 5.375;
ahw = 0.001; % for water contributes like 1/20 of height
bhw = Wc/Weff; 
chw = 20;  
ahf = 0.01;
khf = 1;
Hf = 1/20 * khf/5; % for fertilizer contributes like 1/20 of height

Weff = 5.375 * ones(N, 1); 
Wc = zeros(N, 1);  
for n=24:24:N-24
    Wc(n+1:n+24) = Wc(n-23:n) + Weff(n);
end

Feff = zeros(N, 1);
lastFeff = 0;
m = 24*30;
for n=1:m:N-m
    Feff(n) = 5.375 + lastFeff;
    lastFeff = Feff(n);
end

carrying_not_reached = true;
for n=1:N-1
    if h(n) < kh && carrying_not_reached
        dhdt = ah * h(n) * (1 - h(n)/kh) +...
               Hw * ahw * Weff(n) * exp(-((Wc(n) - bhw * Weff(n))/(chw * Weff(n)))^2) +...
               Hf * ahf * Feff(n) * (1 - Feff(n)/khf);
    elseif h(n) >= kh
        dhdt = ah * h(n) * (1 - h(n)/kh) - dh * h(n) +...
               Hw * ahw * Weff(n) * exp(-((Wc(n) - bhw * Weff(n))/(chw * Weff(n)))^2) +...
               Hf * ahf * Feff(n) * (1 - Feff(n)/khf);
        carrying_not_reached = false;
    else
        dhdt = ah * h(n) * (1 - h(n)/kh) - dh * h(n) +...
               Hw * ahw * Weff(n) * exp(-((Wc(n) - bhw * Weff(n))/(chw * Weff(n)))^2) +...
               Hf * ahf * Feff(n) * (1 - Feff(n)/khf);
    end
    h(n+1) = h(n) + dhdt * dt;
end

figure(1)
plot(1:N, h)

%figure(1)
%hold on
%syms h(t);
%h0 = 10; % only use to simulate after carrying capacity reached
%dh = 0;
%ode = diff(h, t) == ah * h(t) * (1 - h(t)/kh) - dh * h(t) + Hw * ahw *...
%                    5.375 * exp(-((500 - bhw * 5.375)/(chw * 5.375))^2) + Hf * ahf * 0.8 * (1 - 0.8/khf);
%cond = h(0) == h0;
%ySol(t) = dsolve(ode, cond);
%fplot(ySol(t), [0, N])
%legend('Location','northwest')

end