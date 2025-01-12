function basic_plot(filename, column_index)

% e.g. run >> basic_plot('output.csv', 1)

% Read the data from the CSV file
data = readtable(filename);

% Extract the data from the specified column
y = data{:, column_index};

% Plot the data
figure;
plot(y);
title(strcat(filename, ' column ', column_index));
xlabel('Index');
ylabel('Value');

end