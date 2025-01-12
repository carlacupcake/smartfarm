function plant_growth_animation()

% Read-in data
growth_results_data = readmatrix('grapes_growth_results_single_plant.csv');
height = growth_results_data(:, 1);
leaf_area = growth_results_data(:, 2);
canopy_biomass = growth_results_data(:, 3);
fruit_biomass = growth_results_data(:, 4);
N = length(height); % hours

% Animation setup
figure
hold on;
axis equal;
xlim([-10, 10]);
ylim([-10, 10]);
zlim([0, 20]);

% Loop through time steps
for t = 1:N
    % Clear previous plot
    cla;

    % Stem (rectangular prism)
    stem_height = h(t);
    [X, Y, Z] = cuboid([-2, 2], [-2, 2], [0, stem_height]);

    % Plot stem
    fill3(X, Y, Z, [0.6 0.4 0.2]); % Brown color for stem

    % Sphere at top of stem
    radius = sqrt(A(t) / (4 * pi)); % Calculate radius from area
    [Xs, Ys, Zs] = sphere(20); % Create a sphere

    % Adjust sphere position and scale
    Xs = Xs * radius;
    Ys = Ys * radius;
    Zs = Zs * radius + stem_height;

    % Determine color based on canopy density
    green_value = 1 - c(t); % Darker green for higher density
    sphere_color = [0, green_value, 0];

    % Plot sphere
    surf(Xs, Ys, Zs, 'FaceColor', sphere_color, 'EdgeColor', 'none');

    % Pause for animation effect
    pause(0.1);
end

% Function to create a cuboid (rectangular prism)
function [X, Y, Z] = cuboid(X_range, Y_range, Z_range)
    [X, Y] = meshgrid(X_range, Y_range);
    Z1 = Z_range(1) * ones(size(X));
    Z2 = Z_range(2) * ones(size(X));
    X = [X; X];
    Y = [Y; Y];
    Z = [Z1; Z2];
end

figure

end