function total_model_param_fitting()

% Common Parameters
Tcrit = 0;
Wcrit = 0;
Fcrit = 0;
hourly_temp_water_fertilizer_data = readmatrix('hourly_temp_water_fertilizer.csv');
hourly_temps = hourly_temp_water_fertilizer_data(:, 1);
hourly_water = hourly_temp_water_fertilizer_data(:, 2);
hourly_frtlz = hourly_temp_water_fertilizer_data(:, 3);
N = length(hourly_temps); % hours
dt = 1; % hours

% figure(1)
% subplot(3, 1, 1)
% plot(1:N, hourly_temps)
% xlabel('Hour index')
% ylabel('Temp. [degC]')
% title('Hourly Values')
% subplot(3, 1, 2)
% plot(1:N, hourly_water)
% xlabel('Hour index')
% ylabel('Water [kg]')
% subplot(3, 1, 3)
% plot(1:N, hourly_frtlz)
% xlabel('Hour index')
% ylabel('Fertilizer [kg]')

% Get effective and cumulative values
Topt = 23; % 25
Wopt = 0.01;
omegaT = 1;
omegaW = 1;

Teff = zeros(1, N);
Weff = zeros(1, N);
Feff = zeros(1, N);

Tc = zeros(1, N);
Wc = zeros(1, N);
Fc = zeros(1, N);
for i=1:24:N-23
    %disp(strcat('(', num2str(i), ', ', num2str(i+23), ')'))
    todays_eff_temp = mean(hourly_temps(i:i+23) - Tcrit);
    Teff(i:i+23) = todays_eff_temp;
    
    todays_eff_water = mean(hourly_water(i:i+23) - Wcrit);
    Weff(i:i+23) = todays_eff_water;

    todays_eff_fert = mean(hourly_frtlz(i:i+23) - Fcrit);
    Feff(i:i+23) = todays_eff_fert;

    if i > 24
        Tc(i:i+23) = Tc(i-24:i-1) + todays_eff_temp * dt;
        Wc(i:i+23) = Wc(i-24:i-1) + todays_eff_water * dt;
        Fc(i:i+23) = Fc(i-24:i-1) + todays_eff_fert * dt;
    end
end
TSL = 1 - omegaT*abs(1-Teff/Topt);
WSL = 1 - omegaW*abs(1-Weff/Wopt);
TSP = 1 - abs(1 - 1./TSL);

% figure(2)
% subplot(3, 1, 1)
% plot(1:N, Teff)
% xlabel('Hour index')
% ylabel('Temp. [degC]')
% title('Effective Values')
% subplot(3, 1, 2)
% plot(1:N, Weff)
% xlabel('Hour index')
% ylabel('Water [kg]')
% subplot(3, 1, 3)
% plot(1:N, Feff)
% xlabel('Hour index')
% ylabel('Fertilizer [kg]')

% figure(3)
% subplot(3, 1, 1)
% plot(1:N, Tc)
% xlabel('Hour index')
% ylabel('Temp. [degC]')
% title('Cumulative Values')
% subplot(3, 1, 2)
% plot(1:N, Wc)
% xlabel('Hour index')
% ylabel('Water [kg]')
% subplot(3, 1, 3)
% plot(1:N, Fc)
% xlabel('Hour index')
% ylabel('Fertilizer [kg]')

% Initial Conditions
h0 = 0.01;
A0 = 1e-5;
c0 = 0.01;
P0 = 0.01;

% Initialize Time-Dependent Variables
h = h0*ones(N, 1);
A = A0*ones(N, 1);
c = c0*ones(N, 1);
P = P0*ones(N, 1);

% Plant Height Parameters
ah = 0.008; 
kh = 1; 
dh = 0.005; 
Hw = 0.02; 
ahw = 0.001; 
bhw = 100; 
chw = 20;  
ahf = 0.01;
khf = 1;
Hf = 0.01; 

% Leaf Area Parameters
kL = 15e-4; % leaf carrying capacity m2
bLT = 65; 
cLT = 20; 
LT = 5e-5; % goal is 15 cm2 per 30 days or 5e-5 m2 per day
aLT = 25e-4; 
dLT = 5e-4;
bLW = 37; 
cLW = 20; 
Lw = 1e-2;
aLw = 8e-4;
dLW = 1e-4; 
kLF = 0.7;
Lf = 1e-3;
aLf = 1e-4; 

% Canopy Biomass Parameters
ce = 0.05;
R0 = 2045;
kappa = sin(pi/8);
rhostd = 1;
kc = 1;
dc = 5e-3;

% Fruit Biomass Parameters
cthreshold_fruit = 3/4 * kc;
kP = 5; % kg
ap = 50; 
dp = 0.001;

% Simulation
height_decay_triggered = false;
leaf_decay_triggered = false;
for n=1:N-1

    % Plant Height Update
    if h(n) > kh
        height_decay_triggered = true;
    end
    if height_decay_triggered
        dhdt = ah * h(n) * (1 - h(n)/kh)...
               + Hw * ahw * Weff(n) * exp(-((Wc(n) - bhw * Weff(n))/(chw * Weff(n)))^2)...
               + Hf * ahf * Feff(n) * (1 - Feff(n)/khf)...
               - dh * h(n);
    else
        dhdt = ah * h(n) * (1 - h(n)/kh)...
               + Hw * ahw * Weff(n) * exp(-((Wc(n) - bhw * Weff(n))/(chw * Weff(n)))^2)...
               + Hf * ahf * Feff(n) * (1 - Feff(n)/khf);
    end
    h(n+1) = h(n) + dhdt * dt;

    % Leaf Area Update
    if A(n) > kL
        leaf_decay_triggered = true;
    end
    if leaf_decay_triggered
        dAdt = LT * aLT * Teff(n) * exp(-((Tc(n) - bLT*Teff(n))/(cLT*Teff(n)))^2)...
               + Lw * aLw * Weff(n) * exp(-((Wc(n) - bLW*Weff(n))/(cLW*Weff(n)))^2)...
               + Lf * aLf * Feff(n) * (1 - Feff(n)/kLF)...
               - dLT * TSL(n) * A(n) - dLW * WSL(n) * A(n);
    else
        dAdt = LT * aLT * Teff(n) * exp(-((Tc(n) - bLT*Teff(n))/(cLT*Teff(n)))^2)...
               + Lw * aLw * Weff(n) * exp(-((Wc(n) - bLW*Weff(n))/(cLW*Weff(n)))^2)...
               + Lf * aLf * Feff(n) * (1 - Feff(n)/kLF);
    end
    A(n+1) = A(n) + dAdt * dt;

    % Canopy Biomass Update
    if leaf_decay_triggered
        dcdt = ce * R0 * (1 - exp(-kappa/rhostd * 3 * c(n)/A(n) * sqrt(pi/A(n)) * tan(pi/4)))...
               * A(n) * c(n) * (1 - c(n)/kc) - dc*c(n);
    else
        dcdt = ce * R0 * (1 - exp(-kappa/rhostd * 3 * c(n)/A(n) * sqrt(pi/A(n)) * tan(pi/4)))...
               * A(n) * c(n) * (1 - c(n)/kc);
    end    
    c(n+1) = c(n) + dcdt * dt;

    % Fruit Biomass Update
    if P(n) >= kP
        leaf_decay_triggered = true;
    end
    if leaf_decay_triggered
        dPdt = ap * TSP(n) * dcdt * P(n) * (1 - P(n)/kP * kc/c(n))- dp*P(n);
    elseif c(n) < cthreshold_fruit
        dPdt = 0;
    else
        dPdt = ap * TSP(n) * dcdt * P(n) * (1 - P(n)/kP * kc/c(n));
    end
    P(n+1) = P(n) + dPdt * dt;

end

figure(4)
subplot(4, 1, 1)
plot(1:N, h)
xlabel('Hour index')
ylabel('Plant Height (m)')
title('Components of Crop Growth')
subplot(4, 1, 2)
plot(1:N, A) 
xlabel('Hour index')
ylabel('Leaf Area (m2)')
subplot(4, 1, 3)
plot(1:N, c) 
xlabel('Hour index')
ylabel('Canopy Biomass (kg)')
subplot(4, 1, 4)
plot(1:N, P) 
xlabel('Hour index')
ylabel('Fruit Biomass (kg)')

end