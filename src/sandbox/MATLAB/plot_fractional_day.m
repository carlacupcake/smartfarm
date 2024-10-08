function plot_fractional_day()

% Inputs
d  = 1:365;  % integral days of the year
hs = 1:4:24; % choose any hour, does not affect things too much

% Outputs
for h=hs
    gamma = 2*pi/365* (d - 1 + (h - 12)/24);

    figure(1)
    hold on
    plot(d, gamma, 'LineWidth', 3, 'DisplayName', num2str(h))
    
    disp(min(gamma))
    disp(max(gamma))
end

figure(1)
xlabel('Hour of the day', 'FontSize', 20, 'Interpreter', 'latex')
ylabel('Fractional Day', 'FontSize', 20, 'Interpreter', 'latex')
title('How Fractional Day of the Year Changes through the Day/Year',...
      'FontSize', 20, 'Interpreter', 'latex')
xlim([1, 365])

end