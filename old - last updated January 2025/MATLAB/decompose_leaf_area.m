function decompose_leaf_area()

% Common Parameters
Tcrit = 0;
Wcrit = 0;
Fcrit = 0;

hourly_temp_water_fertilizer_data = readmatrix('../hourly_temp_water_fertilizer.csv');

hourly_temps = hourly_temp_water_fertilizer_data(:, 1);
hourly_water = hourly_temp_water_fertilizer_data(:, 2);
hourly_frtlz = hourly_temp_water_fertilizer_data(:, 3);

hourly_temps = [hourly_temps; hourly_temps];
hourly_water = [hourly_water; hourly_water];
hourly_frtlz = [hourly_frtlz; hourly_frtlz];

hourly_temps = 23 * ones(size(hourly_temps));
hourly_water = 0.01 * ones(size(hourly_water));
hourly_frtlz = 0.01 * ones(size(hourly_frtlz));

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
A0 = 1e-5 * 1/3;

% Initialize Time-Dependent Variables
h = h0*ones(N, 1);
A = zeros(N, 1);
A_temp  = A0*ones(N, 1);
A_water = A0*ones(N, 1);
A_fert  = A0*ones(N, 1);

% Leaf Area Parameters
bLT = 65; 
cLT = 20; 
LT = 2.5e-6; 
aLT = 8e-2; 
bLW = 37; 
cLW = 20; 
Lw = 6e-3; 
aLw = 8e-2;
kLF = 0.7;
Lf = 1e-3;
aLf = 8e-4;

% Simulation
for n=1:N-1

    % Leaf Area Growth Terms
    dAdt_temp = LT * aLT * Teff(n) * exp(-((Tc(n) - bLT*Teff(n))/(cLT*Teff(n)))^2);
    dAdt_water = Lw * aLw * Weff(n) * exp(-((Wc(n) - bLW*Weff(n))/(cLW*Weff(n)))^2);
    dAdt_fert = Lf * aLf * Feff(n) * (1 - Feff(n)/kLF);

    A_temp(n+1)  = A_temp(n)  + dAdt_temp  * dt;
    A_water(n+1) = A_water(n) + dAdt_water * dt;
    A_fert(n+1)  = A_fert(n)  + dAdt_fert  * dt;
    A(n+1) = A_temp(n+1) + A_water(n+1) + A_fert(n+1);

end

figure(4)
hold on
plot(1:N, A_temp, 'DisplayName', 'Temp. contrib.')
plot(1:N, A_water, 'DisplayName', 'Water contrib.')
plot(1:N, A_fert, 'DisplayName', 'Fert. contrib.')
plot(1:N, A, 'DisplayName', 'Total leaf area')
xlabel('Hour index')
ylabel('Leaf area (m2)')
title('Decomposing contributions to leaf area growth')
legend()

end