function plot_farm_sim_inputs(filename)

% Run as >> plot_farm_sim_inputs('farm_sim_inputs.csv')

%% Column data assumed to be
% 1  hourly_temperatures
% 2  hourly_irrigation
% 3  hourly_fertilizer
% 4  effective_temperatures
% 5  effective_irrigation
% 6  effective_fertilizer
% 7  cumulative_temperatures
% 8  cumulative_irrigation
% 9  cumulative_fertilizer
% 10 leaf_sensitivity_temp
% 11 fruit_sensitivity_temp
% 12 leaf_sensitivity_water

%% Read the data from the CSV file
data = readtable(filename);

%% Extract desired data
num_hours = size(data, 1);
num_days = ceil(num_hours/24); % 24 hours per day
hours = 1:num_hours;
days  = 1:num_days;

hourly_temperature = data{:, 1};
hourly_irrigation  = data{:, 2};
hourly_fertilizer  = data{:, 3};

effective_temperature_hourly = data{:, 4};
effective_irrigation_hourly  = data{:, 5};
effective_fertilizer_hourly  = data{:, 6};
effective_temperature = effective_temperature_hourly(1:24:end);
effective_irrigation  = effective_irrigation_hourly(1:24:end);
effective_fertilizer  = effective_fertilizer_hourly(1:24:end);

cumulative_temperature_hourly = data{:, 7};
cumulative_irrigation_hourly  = data{:, 8};
cumulative_fertilizer_hourly  = data{:, 9};
cumulative_temperature = cumulative_temperature_hourly(1:24:end);
cumulative_irrigation  = cumulative_irrigation_hourly(1:24:end);
cumulative_fertilizer  = cumulative_fertilizer_hourly(1:24:end);

leaf_sensitivity_temp_hourly  = data{:, 10};
fruit_sensitivity_temp_hourly = data{:, 11};
leaf_sensitivity_water_hourly = data{:, 12};
leaf_sensitivity_temp  = leaf_sensitivity_temp_hourly(1:24:end);
fruit_sensitivity_temp = fruit_sensitivity_temp_hourly(1:24:end);
leaf_sensitivity_water = leaf_sensitivity_water_hourly(1:24:end);

%% Plot Settings
% Get the screen size
screenSize = get(0, 'ScreenSize'); % [left, bottom, width, height]

% Define figure dimensions for the left half of the screen
figureWidth = screenSize(3)/2;  % Half the screen width
figureHeight = screenSize(4);   % Full screen height
figureLeft = 0;                 % Start from the left edge
figureBottom = 0;               % Start from the bottom edge

% Set default font sizes for different plot elements
set(groot, 'defaultAxesFontSize', 14);         % Font size for axes
set(groot, 'defaultTextFontSize', 16);         % Font size for titles and text
set(groot, 'defaultLegendFontSize', 12);       % Font size for legends
set(groot, 'defaultColorbarFontSize', 12);     % Font size for colorbars

% Set the default interpreter to LaTeX for various plot elements
set(groot, 'defaultTextInterpreter', 'latex');           % For text objects
set(groot, 'defaultAxesTickLabelInterpreter', 'latex');  % For axis tick labels
set(groot, 'defaultLegendInterpreter', 'latex');         % For legends
set(groot, 'defaultColorbarTickLabelInterpreter', 'latex'); % For colorbars

%% Plot the hourly temperature, irrigation, and fertilizer data
figure('Position', [figureLeft, figureBottom, figureWidth, figureHeight]);

% Hourly Temperature
subplot(3, 1, 1);
plot(hours, hourly_temperature, 'Color', [0, 0.5, 0], 'LineWidth', 2);
xlabel('Index');
ylabel('Temperature (degC)');
title('Hourly Temperature');

% Hourly Irrigation
subplot(3, 1, 2);
plot(hours, hourly_irrigation, 'Color', [0, 0.5, 0], 'LineWidth', 2);
xlabel('Index');
ylabel('Consumption (kg)');
title('Hourly Water Consumption');

% Hourly Fertilizer
subplot(3, 1, 3);
plot(hours, hourly_fertilizer, 'Color', [0, 0.5, 0], 'LineWidth', 2);
xlabel('Index');
ylabel('Consumption (kg)');
title('Hourly Fertilizer Consumption');

%% Plot the daily effective inputs
figure('Position', [figureLeft, figureBottom, figureWidth, figureHeight]);

% Daily Effective Temperature
subplot(3, 1, 1);
plot(days, effective_temperature, 'Color', [0, 0.5, 0], 'LineWidth', 2);
xlabel('Index');
ylabel('Temperature (degC)');
title('Daily Effective Temperature');

% Daily Effective Irrigation
subplot(3, 1, 2);
plot(days, effective_irrigation, 'Color', [0, 0.5, 0], 'LineWidth', 2);
xlabel('Index');
ylabel('Water for Irrigation (kg)');
title('Daily Effective Irrigation');

% Daily Effective Fertilizer
subplot(3, 1, 3);
plot(days, effective_fertilizer, 'Color', [0, 0.5, 0], 'LineWidth', 2);
xlabel('Index');
ylabel('Fertilizer Deposited (kg)');
title('Daily Effective Fertilizer');

%% Plot the daily cumulative inputs
figure('Position', [figureLeft, figureBottom, figureWidth, figureHeight]);

% Daily Cumulative Temperature
subplot(3, 1, 1);
plot(days, cumulative_temperature, 'Color', [0, 0.5, 0], 'LineWidth', 2);
xlabel('Index');
ylabel('Temperature (degC-hour)');
title('Daily Cumulative Temperature');

% Daily Cumulative Irrigation
subplot(3, 1, 2);
plot(days, cumulative_irrigation, 'Color', [0, 0.5, 0], 'LineWidth', 2);
xlabel('Index');
ylabel('Water for Irrigation (kg-hour)');
title('Daily Cumulative Irrigation');

% Daily Cumulative Fertilizer
subplot(3, 1, 3);
plot(days, cumulative_fertilizer, 'Color', [0, 0.5, 0], 'LineWidth', 2);
xlabel('Index');
ylabel('Fertilizer Deposited (kg-hour)');
title('Daily Cumulative Fertilizer');

%% Plot the daily sensitivity inputs
figure('Position', [figureLeft, figureBottom, figureWidth, figureHeight]);

% Daily Cumulative Temperature
subplot(3, 1, 1);
plot(days, leaf_sensitivity_temp, 'Color', [0, 0.5, 0], 'LineWidth', 2);
xlabel('Index');
ylabel('Unitless (1 =  neutral)');
title('Leaf Sensitivity to Temperature');

% Daily Cumulative Irrigation
subplot(3, 1, 2);
plot(days, fruit_sensitivity_temp, 'Color', [0, 0.5, 0], 'LineWidth', 2);
xlabel('Index');
ylabel('Unitless (1 =  neutral)');
title('Fruit Sensitivity to Temperature');

% Daily Cumulative Fertilizer
subplot(3, 1, 3);
plot(days, leaf_sensitivity_water, 'Color', [0, 0.5, 0], 'LineWidth', 2);
xlabel('Index');
ylabel('Unitless (1 =  neutral)');
title('Leaf Sensitivity to Irrigation');

end