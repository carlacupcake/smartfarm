function gaussian_like_fertilizer()

% Common Parameters
Fcrit = 0; 

hourly_temp_water_fertilizer_data = readmatrix('../hourly_temp_water_fertilizer.csv');
hourly_fert = hourly_temp_water_fertilizer_data(:, 3);
hourly_fert = [hourly_fert; hourly_fert];
hourly_fert = 23 * ones(size(hourly_fert));

N = length(hourly_fert); % hours
dt = 1; % hours

% Get effective and cumulative values
%Teff = zeros(1, N);
%Weff = zeros(1, N);
Feff = zeros(1, N);

%Tc = zeros(1, N);
%Wc = zeros(1, N);
Fc = zeros(1, N);
for i=1:24:N-23

    %todays_eff_temp = mean(hourly_temps(i:i+23) - Tcrit);
    %Teff(i:i+23) = todays_eff_temp;
    
    %todays_eff_water = mean(hourly_fert(i:i+23) - Wcrit);
    %Weff(i:i+23) = todays_eff_water;

    todays_eff_fert = mean(hourly_fert(i:i+23) - Fcrit);
    Feff(i:i+23) = todays_eff_fert;

    if i > 24
        %Tc(i:i+23) = Tc(i-24:i-1) + todays_eff_temp * dt;
        %Wc(i:i+23) = Wc(i-24:i-1) + todays_eff_water * dt;
        Fc(i:i+23) = Fc(i-24:i-1) + todays_eff_fert * dt;
    end
end

% Initial Conditions
%A0 = 1e-5;
A0 = 0.01;

% Initialize Time-Dependent Variables
A = A0 * ones(N, 1);

% Leaf Area Parameters
%aLFs = [6, 6.5, 7, 7.5, 8, 8.5, 9, 9.5, 10] * 1e-3; %aLF = 8e-3; 
%bLFs = [1, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200]; %bLF = 65; 
%cLFs = [5, 10, 15, 20, 25, 30, 35, 40]; %cLF = 20;  
%LFs = [4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8] * 1e-6; %LF = 6e-6;

aLFs = [6, 6.5, 7, 7.5, 8, 8.5, 9, 9.5, 10] * 1e-2; %aHF = 0.075
bLFs = [1, 10, 20, 40, 60, 80, 100, 120, 140, 160, 180]; %bHF = 80 
cLFs = [1, 20, 25, 30, 35, 40, 45]; %cHF = 35
LFs = [2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6] * 1e-5; %HF = 4e-5

%% Initialize plot for aLF
figure(1)
hold on

% Simulations for aLF
for aLF=aLFs
    bLF = bLFs(6);
    cLF = cLFs(6);
    LF = LFs(6);
    for n=1:N-1    
        dAdt = LF * aLF * Feff(n) * exp(-((Fc(n) - bLF*Feff(n))/(cLF*Feff(n)))^2);
        A(n+1) = A(n) + dAdt * dt;
    end

    figure(1)
    plot(1:N, A, 'LineWidth', 3, 'DisplayName', strcat('$a_{LF} = ', num2str(aLF), '$'));

end

figure(1)
xlabel('Hour index', 'Interpreter', 'latex', 'FontSize', 18)
ylabel('Leaf area (m$^2$)', 'Interpreter', 'latex', 'FontSize', 18)
title('Effect of $a_{LF}$ on Leaf Area Growth', 'Interpreter', 'latex', 'FontSize', 20)
legend('Interpreter', 'latex', 'FontSize', 15, 'Location', 'southeast')
%saveas(gcf, 'effect_aLF_leaf_area.png');

%% Initialize plot for bLF
figure(2)
hold on

% Simulations for bLF
for bLF=bLFs
    aLF = aLFs(6);
    cLF = cLFs(6);
    LF = LFs(6);
    for n=1:N-1    
        dAdt = LF * aLF * Feff(n) * exp(-((Fc(n) - bLF*Feff(n))/(cLF*Feff(n)))^2);
        A(n+1) = A(n) + dAdt * dt;
    end

    figure(2)
    plot(1:N, A, 'LineWidth', 3, 'DisplayName', strcat('$k_{LF} = ', num2str(bLF), '$'));

end

figure(2)
xlabel('Hour index', 'Interpreter', 'latex', 'FontSize', 18)
ylabel('Leaf area (m$^2$)', 'Interpreter', 'latex', 'FontSize', 18)
title('Effect of $b_{LF}$ on Leaf Area Growth', 'Interpreter', 'latex', 'FontSize', 20)
legend('Interpreter', 'latex', 'FontSize', 15, 'Location', 'southeast')
%saveas(gcf, 'effect_bLF_leaf_area.png');

%% Initialize plot for cLF
figure(3)
hold on

% Simulations for LF
for cLF=cLFs
    aLF = aLFs(6);
    bLF = bLFs(6);
    LF = LFs(6);
    for n=1:N-1    
        dAdt = LF * aLF * Feff(n) * exp(-((Fc(n) - bLF*Feff(n))/(cLF*Feff(n)))^2);
        A(n+1) = A(n) + dAdt * dt;
    end

    figure(3)
    plot(1:N, A, 'LineWidth', 3, 'DisplayName', strcat('$c_{LF} = ', num2str(cLF), '$'));

end

figure(3)
xlabel('Hour index', 'Interpreter', 'latex', 'FontSize', 18)
ylabel('Leaf area (m$^2$)', 'Interpreter', 'latex', 'FontSize', 18)
title('Effect of $c_{LF}$ on Leaf Area Growth', 'Interpreter', 'latex', 'FontSize', 20)
legend('Interpreter', 'latex', 'FontSize', 15, 'Location', 'southeast')
%saveas(gcf, 'effect_cLF_leaf_area.png');

%% Initialize plot for LF
figure(4)
hold on

% Simulations for LF
for LF=LFs
    aLF = aLFs(6);
    bLF = bLFs(6);
    cLF = cLFs(6);
    for n=1:N-1    
        dAdt = LF * aLF * Feff(n) * exp(-((Fc(n) - bLF*Feff(n))/(cLF*Feff(n)))^2);
        A(n+1) = A(n) + dAdt * dt;
    end

    figure(4)
    plot(1:N, A, 'LineWidth', 3, 'DisplayName', strcat('$L_{F} = ', num2str(LF), '$'));

end

figure(4)
xlabel('Hour index', 'Interpreter', 'latex', 'FontSize', 18)
ylabel('Leaf area (m$^2$)', 'Interpreter', 'latex', 'FontSize', 18)
title('Effect of $L_{F}$ on Leaf Area Growth', 'Interpreter', 'latex', 'FontSize', 20)
legend('Interpreter', 'latex', 'FontSize', 15, 'Location', 'southeast')
%saveas(gcf, 'effect_LF_leaf_area.png');

end