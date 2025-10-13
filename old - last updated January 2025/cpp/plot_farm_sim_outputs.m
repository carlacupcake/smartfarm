function plot_farm_sim_outputs(filename)

% Run as >> plot_farm_sim_outputs('farm_sim_outputs.csv')

%% Column data assumed to be
% 1 height
% 2 leaf_area
% 3 canopy_biomass
% 4 fruit_biomass

%% Read the data from the CSV file
data = readtable(filename);

%% Extract desired data
num_hours = size(data, 1);
hours = 1:num_hours;

height         = data{:, 1};
leaf_area      = data{:, 2};
canopy_biomass = data{:, 3};
fruit_biomass  = data{:, 4};

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

%% Plot the state variable evolution over time
figure('Position', [figureLeft, figureBottom, figureWidth, figureHeight]);

% Plant Height
subplot(4, 1, 1);
plot(hours, height, 'Color', [0, 0.5, 0], 'LineWidth', 2);
xlabel('Index');
ylabel('Height (m)');
title('Plant Height');

% Leaf Area
subplot(4, 1, 2);
plot(hours, leaf_area, 'Color', [0, 0.5, 0], 'LineWidth', 2);
xlabel('Index');
ylabel('Area (m2)');
title('Leaf Area');

% Canopy Biomass
subplot(4, 1, 3);
plot(hours, canopy_biomass, 'Color', [0, 0.5, 0], 'LineWidth', 2);
xlabel('Index');
ylabel('Biomass (kg)');
title('Canopy Biomass');

% Fruit Biomass
subplot(4, 1, 4);
plot(hours, fruit_biomass, 'Color', [0, 0.5, 0], 'LineWidth', 2);
xlabel('Index');
ylabel('Biomass (kg)');
title('Fruit Biomass');

end