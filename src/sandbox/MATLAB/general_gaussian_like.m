function general_gaussian_like()

%% Plotting color palette
greens = [138, 237, 140,
          103, 209, 105,
          69,  182, 71,
          34,  155, 36,
          0,   128, 2,
          0,   112, 1,
          0,   96,  1,
          0,   80,  1,
          0,   64,  1]/255;

%% Common Parameters
Wcrit = 0; 

hourly_temp_water_fertilizer_data = readmatrix('../hourly_temp_water_fertilizer.csv');
hourly_water = hourly_temp_water_fertilizer_data(:, 2);
hourly_water = [hourly_water; hourly_water];
hourly_water = 23 * ones(size(hourly_water));

N = length(hourly_water); % hours
dt = 1; % hours

% Get effective and cumulative values
%Teff = zeros(1, N);
Weff = zeros(1, N);
%Feff = zeros(1, N);

%Tc = zeros(1, N);
Wc = zeros(1, N);
%Fc = zeros(1, N);
for i=1:24:N-23

    %todays_eff_temp = mean(hourly_temps(i:i+23) - Tcrit);
    %Teff(i:i+23) = todays_eff_temp;
    
    todays_eff_water = mean(hourly_water(i:i+23) - Wcrit);
    Weff(i:i+23) = todays_eff_water;

    %todays_eff_fert = mean(hourly_frtlz(i:i+23) - Fcrit);
    %Feff(i:i+23) = todays_eff_fert;

    if i > 24
        %Tc(i:i+23) = Tc(i-24:i-1) + todays_eff_temp * dt;
        Wc(i:i+23) = Wc(i-24:i-1) + todays_eff_water * dt;
        %Fc(i:i+23) = Fc(i-24:i-1) + todays_eff_fert * dt;
    end
end

% Initial Conditions
%A0 = 1e-5;
A0 = 0.01;

% Initialize Time-Dependent Variables
A = A0 * ones(N, 1);

% Leaf Area Parameters
%aLWs = [6, 6.5, 7, 7.5, 8, 8.5, 9, 9.5, 10] * 1e-3; %aLW = 8e-3; 
%bLWs = [1, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200]; %bLW = 37; 
%cLWs = [5, 10, 15, 20, 25, 30, 35, 40]; %cLW = 20;  
%Lws = [4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8] * 1e-6; %LW = 6e-6;

aLWs = [5.5, 6, 6.5, 7, 7.5, 8, 8.5, 9, 9.5] * 1e-2; %aHW = 0.075
bLWs = [10, 20, 40, 60, 80, 100, 120, 140, 160]; %bHW = 80 
cLWs = [15, 20, 25, 30, 35, 40, 45, 50, 55]; %cHW = 35
Lws = [2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6] * 1e-5; %HW = 4e-5

%% Initialize plot for aLW
figure(1)
hold on
colororder(greens)

% Simulations for aLW
for aLW=aLWs
    bLW = bLWs(5);
    cLW = cLWs(5);
    Lw = Lws(5);
    for n=1:N-1    
        dAdt = Lw * aLW * Weff(n) * exp(-((Wc(n) - bLW*Weff(n))/(cLW*Weff(n)))^2);
        A(n+1) = A(n) + dAdt * dt;
    end

    figure(1)
    plot(1:N, A, 'LineWidth', 3, 'DisplayName', strcat('$a_{LW} = ', num2str(aLW), '$'));

end

figure(1)
xlabel('t', 'Interpreter', 'latex', 'FontSize', 18)
ylabel('$A(t)$', 'Interpreter', 'latex', 'FontSize', 18)
title('Effect of $a$ on simple Gaussian-like growth', 'Interpreter', 'latex', 'FontSize', 20)
%legend('Interpreter', 'latex', 'FontSize', 15, 'Location', 'southeast')
set(gca, 'XTickLabel', [], 'YTickLabel', []);
colormap(greens);
colorbar;
exportgraphics(gca, 'effect_a_general.png');

%% Initialize plot for bLW
figure(2)
hold on
colororder(greens)

% Simulations for bLW
for bLW=bLWs
    aLW = aLWs(5);
    cLW = cLWs(5);
    Lw = Lws(5);
    for n=1:N-1    
        dAdt = Lw * aLW * Weff(n) * exp(-((Wc(n) - bLW*Weff(n))/(cLW*Weff(n)))^2);
        A(n+1) = A(n) + dAdt * dt;
    end

    figure(2)
    plot(1:N, A, 'LineWidth', 3, 'DisplayName', strcat('$b_{LW} = ', num2str(bLW), '$'));

end

figure(2)
xlabel('t', 'Interpreter', 'latex', 'FontSize', 18)
ylabel('$A(t)$', 'Interpreter', 'latex', 'FontSize', 18)
title('Effect of $b$ on simple Gaussian-like growth', 'Interpreter', 'latex', 'FontSize', 20)
%legend('Interpreter', 'latex', 'FontSize', 15, 'Location', 'southeast')
set(gca, 'XTickLabel', [], 'YTickLabel', []);
colormap(greens);
colorbar;
exportgraphics(gca, 'effect_b_general.png');

%% Initialize plot for cLW
figure(3)
hold on
colororder(greens)

% Simulations for cLW
for cLW=cLWs
    aLW = aLWs(5);
    bLW = bLWs(5);
    Lw = Lws(5);
    for n=1:N-1    
        dAdt = Lw * aLW * Weff(n) * exp(-((Wc(n) - bLW*Weff(n))/(cLW*Weff(n)))^2);
        A(n+1) = A(n) + dAdt * dt;
    end

    figure(3)
    plot(1:N, A, 'LineWidth', 3, 'DisplayName', strcat('$c_{LW} = ', num2str(cLW), '$'));

end

figure(3)
xlabel('t', 'Interpreter', 'latex', 'FontSize', 18)
ylabel('$A(t)$', 'Interpreter', 'latex', 'FontSize', 18)
title('Effect of $c$ on simple Gaussian-like growth', 'Interpreter', 'latex', 'FontSize', 20)
%legend('Interpreter', 'latex', 'FontSize', 15, 'Location', 'southeast')
set(gca, 'XTickLabel', [], 'YTickLabel', []);
colormap(greens);
colorbar;
exportgraphics(gca, 'effect_c_general.png');

%% Initialize plot for LW
figure(4)
hold on
colororder(greens)

% Simulations for LT
for Lw=Lws
    aLW = aLWs(5);
    bLW = bLWs(5);
    cLW = cLWs(5);
    for n=1:N-1    
        dAdt = Lw * aLW * Weff(n) * exp(-((Wc(n) - bLW*Weff(n))/(cLW*Weff(n)))^2);
        A(n+1) = A(n) + dAdt * dt;
    end

    figure(4)
    plot(1:N, A, 'LineWidth', 3, 'DisplayName', strcat('$L_{W} = ', num2str(Lw), '$'));

end

figure(4)
xlabel('t', 'Interpreter', 'latex', 'FontSize', 18)
ylabel('$A(t)$', 'Interpreter', 'latex', 'FontSize', 18)
title('Effect of gain term on simple Gaussian-like growth', 'Interpreter', 'latex', 'FontSize', 20)
%legend('Interpreter', 'latex', 'FontSize', 15, 'Location', 'southeast')
set(gca, 'XTickLabel', [], 'YTickLabel', []);
colormap(greens);
colorbar;
exportgraphics(gca, 'effect_gain_general.png');

end