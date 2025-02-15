// utilities.h
#ifndef UTILITIES_H
#define UTILITIES_H

#include <string>
#include <vector>

// Function to read solar radiation and temperature data from input CSV
std::vector<double> load_data_from_csv_column(const std::string& filename, 
                                              int column_index, 
                                              bool has_header);

void print_data_from_vector(std::vector<double> data, 
                            const std::string& print_string);

std::string trim(const std::string& str);

std::vector<double> load_hourly_inputs(int start_point, 
                                       int end_point, 
                                       int period, 
                                       double amount, 
                                       int total_time_steps);

std::vector<double> calc_avg_daily_values(std::vector<double> hourly_values,
                                          int total_time_steps);

std::vector<double> calc_leaf_sens_values(std::vector<double> average_temperatures,
                                          double critical_value,
                                          double optimal_value,
                                          int total_time_steps);

std::vector<double> calc_fruit_sens_values(std::vector<double> average_temperatures,
                                           std::vector<double> leaf_sensitivity_values,
                                           double temp_critical_fruit_growth,
                                           double temp_ceiling_fruit_growth,
                                           double temp_optimal_fruit_growth,
                                           int total_time_steps);

std::vector<double> calc_eff_values(std::vector<double> hourly_values,
                                    double critical_value,
                                    int total_time_steps);

std::vector<double> calc_cum_values(std::vector<double> effective_values,
                                    int total_time_steps);

void write_vectors_to_csv(const std::string& filename, 
                          const std::vector<std::vector<double>>& vectors);

#endif // UTILITIES_H
