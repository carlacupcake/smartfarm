// utilities.cpp
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include "constants.h"
#include "utilities.h"

std::string trim(const std::string& str) {
    size_t start = str.find_first_not_of(" \t");
    size_t end = str.find_last_not_of(" \t");
    return (start == std::string::npos) ? "" : str.substr(start, end - start + 1);
}

std::vector<double> load_data_from_csv_column(const std::string& filename, 
                                              int column_index, 
                                              bool has_header) {
    std::vector<double> column_data;
    std::ifstream file(filename);

    if (!file.is_open()) {
        std::cerr << "Error opening file: " << filename << std::endl;
        return column_data; // Return empty vector
    }

    std::string line;
    int line_number = 0; // Keep track of the line number

    // Skip header row if present
    if (has_header && std::getline(file, line)) {
        ++line_number; // Increment line number for the header
    }

    // Read each line of the file
    while (std::getline(file, line)) {
        ++line_number; // Increment line number for the current line
        std::stringstream ss(line);  // Create a stringstream to parse the line
        std::string cell;
        int current_column = 0;

        // Iterate over columns in the row
        while (std::getline(ss, cell, ',')) {
            if (current_column == column_index) {
                // Trim the cell to remove leading/trailing whitespace
                cell = trim(cell);
                try {
                    column_data.push_back(std::stod(cell)); // Convert string to double and store it
                } 
                /*
                catch (const std::invalid_argument& e) {
                    std::cerr << "Invalid value at (line, column): (" << line_number 
                              << ", " << column_index + 1 << ") = \"" << cell << "\"" << std::endl;
                }
                */
                catch (const std::invalid_argument& e) {}
                break; // Exit loop after reading the desired column
            }
            ++current_column;
        }
    }

    file.close();
    return column_data;
}

void print_data_from_vector(std::vector<double> data, 
                            const std::string& print_string) {
    std::cout << print_string;
    for (const auto& value : data) {
        std::cout << value << " ";
    }
    std::cout << std::endl;
    return;
}

std::vector<double> load_hourly_inputs(int start_point, 
                                       int end_point, 
                                       int period, 
                                       double amount, 
                                       int total_time_steps) {
    std::vector<double> hourly_inputs(total_time_steps, 0.0);

    for (int point = start_point; point < end_point; point += period) {
        if (point >= 0 && point < total_time_steps) {
            hourly_inputs[point] = amount;
        }
    }
    return hourly_inputs;
}

std::vector<double> calc_eff_values(std::vector<double> hourly_values,
                                    double critical_value,
                                    int total_time_steps) {
    std::vector<double> effective_values(total_time_steps, 0.0);

    for (int point = 0; point < total_time_steps; point += HOURS_PER_DAY) {
        double sum = 0.0;
        for (size_t i = 0; i < HOURS_PER_DAY; ++i) {
            sum += (hourly_values[point + i] - critical_value);
        }
        double todays_eff_value = sum / HOURS_PER_DAY;
        for (size_t i = 0; i < HOURS_PER_DAY; ++i) {
            effective_values[point + i] = todays_eff_value;
        }
    }
    return effective_values;
}

void write_vectors_to_csv(const std::string& filename, 
                          const std::vector<std::vector<double>>& vectors) {
    // Check that all vectors have the same size
    size_t num_rows = vectors[0].size();
    for (const auto& vec : vectors) {
        if (vec.size() != num_rows) {
            std::cerr << "All vectors must have the same length!" << std::endl;
            return;
        }
    }

    // Open a CSV file for writing
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error opening file: " << filename << std::endl;
        return;
    }

    // Write each row of data from the vectors
    for (size_t i = 0; i < num_rows; ++i) {
        for (size_t j = 0; j < vectors.size(); ++j) {
            file << vectors[j][i];
            if (j != vectors.size() - 1) {
                file << ",";  // Add a comma separator, except for the last column
            }
        }
        file << "\n";  // End of line after each row
    }

    // Close the file
    file.close();
    std::cout << "Data written to " << filename << std::endl;
}
