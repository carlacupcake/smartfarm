function gaussian_like_temp()

%% Common Parameters
Tcrit = 0; 

hourly_temp_water_fertilizer_data = readmatrix('../hourly_temp_water_fertilizer.csv');
hourly_temps = hourly_temp_water_fertilizer_data(:, 1);
hourly_temps = [hourly_temps; hourly_temps];
hourly_temps = 23 * ones(size(hourly_temps));

N = length(hourly_temps); % hours
dt = 1; % hours

% Get effective and cumulative values
Teff = zeros(1, N);
%Weff = zeros(1, N);
%Feff = zeros(1, N);

Tc = zeros(1, N);
%Wc = zeros(1, N);
%Fc = zeros(1, N);
for i=1:24:N-23
    %disp(strcat('(', num2str(i), ', ', num2str(i+23), ')'))
    todays_eff_temp = mean(hourly_temps(i:i+23) - Tcrit);
    Teff(i:i+23) = todays_eff_temp;
    
    %todays_eff_water = mean(hourly_water(i:i+23) - Wcrit);
    %Weff(i:i+23) = todays_eff_water;

    %todays_eff_fert = mean(hourly_frtlz(i:i+23) - Fcrit);
    %Feff(i:i+23) = todays_eff_fert;

    if i > 24
        Tc(i:i+23) = Tc(i-24:i-1) + todays_eff_temp * dt;
        %Wc(i:i+23) = Wc(i-24:i-1) + todays_eff_water * dt;
        %Fc(i:i+23) = Fc(i-24:i-1) + todays_eff_fert * dt;
    end
end

% Initial Conditions
A0 = 1e-5;

% Initialize Time-Dependent Variables
A = A0 * ones(N, 1);

% Leaf Area Parameters
aLTs = [6e-2, 6.5e-2, 7e-2, 7.5e-2, 8e-2, 8.5e-2, 9e-2, 9.5e-2, 10e-2]; %aLT = 8e-2; 
bLTs = [1, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200]; %bLT = 65; 
cLTs = [5, 10, 15, 20, 25, 30, 35, 40]; %cLT = 20; 
LTs = [0.5e-6, 1e-6, 1.5e-6, 2e-6, 2.5e-6, 3e-6, 3.5e-6, 4e-6, 4.5e-6]; %LT = 2.5e-6;  

%% Initialize plot for aLT
figure(1)
hold on

% Simulations for aLT
for aLT=aLTs
    bLT = 65;
    cLT = 20;
    LT = 2.5e-6;
    for n=1:N-1    
        dAdt = LT * aLT * Teff(n) * exp(-((Tc(n) - bLT*Teff(n))/(cLT*Teff(n)))^2);
        A(n+1) = A(n) + dAdt * dt;
    end

    figure(1)
    plot(1:N, A, 'LineWidth', 3, 'DisplayName', strcat('$a_{LT} = ', num2str(aLT), '$'));

end

figure(1)
xlabel('Hour index', 'Interpreter', 'latex', 'FontSize', 18)
ylabel('Leaf area (m$^2$)', 'Interpreter', 'latex', 'FontSize', 18)
title('Effect of $a_{LT}$ on Leaf Area Growth', 'Interpreter', 'latex', 'FontSize', 20)
legend('Interpreter', 'latex', 'FontSize', 15, 'Location', 'southeast')
saveas(gcf, 'effect_aLT_leaf_area.png');

%% Initialize plot for bLT
figure(2)
hold on

% Simulations for bLT
for bLT=bLTs
    aLT = 8e-2;
    cLT = 20;
    LT = 2.5e-6;
    for n=1:N-1    
        dAdt = LT * aLT * Teff(n) * exp(-((Tc(n) - bLT*Teff(n))/(cLT*Teff(n)))^2);
        A(n+1) = A(n) + dAdt * dt;
    end

    figure(2)
    plot(1:N, A, 'LineWidth', 3, 'DisplayName', strcat('$b_{LT} = ', num2str(bLT), '$'));

end

figure(2)
xlabel('Hour index', 'Interpreter', 'latex', 'FontSize', 18)
ylabel('Leaf area (m$^2$)', 'Interpreter', 'latex', 'FontSize', 18)
title('Effect of $b_{LT}$ on Leaf Area Growth', 'Interpreter', 'latex', 'FontSize', 20)
legend('Interpreter', 'latex', 'FontSize', 15, 'Location', 'southeast')
saveas(gcf, 'effect_bLT_leaf_area.png');

%% Initialize plot for cLT
figure(3)
hold on

% Simulations for cLT
for cLT=cLTs
    aLT = 8e-2;
    bLT = 65;
    LT = 2.5e-6;
    for n=1:N-1    
        dAdt = LT * aLT * Teff(n) * exp(-((Tc(n) - bLT*Teff(n))/(cLT*Teff(n)))^2);
        A(n+1) = A(n) + dAdt * dt;
    end

    figure(3)
    plot(1:N, A, 'LineWidth', 3, 'DisplayName', strcat('$c_{LT} = ', num2str(cLT), '$'));

end

figure(3)
xlabel('Hour index', 'Interpreter', 'latex', 'FontSize', 18)
ylabel('Leaf area (m$^2$)', 'Interpreter', 'latex', 'FontSize', 18)
title('Effect of $c_{LT}$ on Leaf Area Growth', 'Interpreter', 'latex', 'FontSize', 20)
legend('Interpreter', 'latex', 'FontSize', 15, 'Location', 'southeast')
saveas(gcf, 'effect_cLT_leaf_area.png');

%% Initialize plot for LT
figure(4)
hold on

% Simulations for LT
for LT=LTs
    aLT = 8e-2;
    bLT = 65;
    cLT = 20;
    for n=1:N-1    
        dAdt = LT * aLT * Teff(n) * exp(-((Tc(n) - bLT*Teff(n))/(cLT*Teff(n)))^2);
        A(n+1) = A(n) + dAdt * dt;
    end

    figure(4)
    plot(1:N, A, 'LineWidth', 3, 'DisplayName', strcat('$L_{T} = ', num2str(LT), '$'));

end

figure(4)
xlabel('Hour index', 'Interpreter', 'latex', 'FontSize', 18)
ylabel('Leaf area (m$^2$)', 'Interpreter', 'latex', 'FontSize', 18)
title('Effect of $L_{T}$ on Leaf Area Growth', 'Interpreter', 'latex', 'FontSize', 20)
legend('Interpreter', 'latex', 'FontSize', 15, 'Location', 'southeast')
saveas(gcf, 'effect_LT_leaf_area.png');

end