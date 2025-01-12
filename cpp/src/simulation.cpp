// simulation.cpp
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <locale>
#include <ctime> // For std::tm

#include "constants.h"
#include "datetime.h"
#include "nlohmann/json.hpp"
#include "user_input.h"
#include "utilities.h"

// Define aliases for simplicity
using json = nlohmann::json;

int main() {

    // Store date/time input
    DateTime start_date(start_year, 
                        start_month, 
                        start_day, 
                        start_hour, 
                        start_minute, 
                        start_second);
    DateTime end_date(end_year, 
                      end_month, 
                      end_day, 
                      end_hour, 
                      end_minute, 
                      end_second);
    DateTime water_start_date(water_start_year, 
                              water_start_month, 
                              water_start_day, 
                              water_start_hour, 
                              water_start_minute, 
                              water_start_second);
    DateTime fertilizer_start_date(fertilizer_start_year, 
                                   fertilizer_start_month, 
                                   fertilizer_start_day, 
                                   fertilizer_start_hour, 
                                   fertilizer_start_minute, 
                                   fertilizer_start_second);

    // Verify input dates/times
    std::tm start_tm            = start_date.toTm();
    std::tm end_tm              = end_date.toTm();
    std::tm water_start_tm      = water_start_date.toTm();
    std::tm fertilizer_start_tm = fertilizer_start_date.toTm();

    std::cout << "Simulation start date and time: " << std::asctime(&start_tm);
    std::cout << "Simulation end date and time: "   << std::asctime(&end_tm);
    std::cout << "Irrigation start date and time: " << std::asctime(&water_start_tm);
    std::cout << "Fertilizer start date and time: " << std::asctime(&fertilizer_start_tm);

    // Calculate NSRDB equivalent points
    int start_point = start_date.get_nsrdb_equiv_point();
    int end_point   = end_date.get_nsrdb_equiv_point();

    // Output the result (you can change this to store or process as needed)
    std::cout << "Start NSRDB point: " << start_point << std::endl;
    std::cout << "End NSRDB point: "   << end_point << std::endl;

    // Get total simulation days and total simulation time steps
    int total_simulation_days = end_date.get_number_day() - start_date.get_number_day();
    int total_time_steps      = total_simulation_days * num_time_steps_per_day - 1;

    // Read-in input solar radiation and temperature data (no header) from CSV
    std::string input_filename   = "hourly_temp_rad.csv";
    int radiation_column_index   = 0;
    int temperature_column_index = 1;
    bool has_header              = false;

    std::vector<double> hourly_radiation = load_data_from_csv_column(input_filename, 
                                                                     radiation_column_index, 
                                                                     has_header);
    std::vector<double> hourly_temperature = load_data_from_csv_column(input_filename, 
                                                                       temperature_column_index, 
                                                                       has_header);
    hourly_temperature.erase(hourly_temperature.begin());

    // Print the extracted data
    //print_data_from_vector(hourly_radiation,   "Radiation Data: ");
    //print_data_from_vector(hourly_temperature, "Temperature Data: ");

    // Set hourly irrigation and fertilizer inputs
    int water_start_point = 0;
    int water_end_point = total_time_steps;
    std::vector<double> hourly_irrigation = load_hourly_inputs(water_start_point, 
                                                               water_end_point, 
                                                               water_period, 
                                                               water_amount_per_plant, 
                                                               total_time_steps);
    int fertilizer_start_point = 0;
    int fertilizer_end_point = total_time_steps;
    std::vector<double> hourly_fertilizer = load_hourly_inputs(fertilizer_start_point, 
                                                               fertilizer_end_point, 
                                                               fertilizer_period, 
                                                               fertilizer_amount_per_plant, 
                                                               total_time_steps);

    // Calculate effective temperatures over simulation length
    std::vector<double> effective_temperatures = calc_eff_values(hourly_temperature,
                                                                 temp_critical_fruit_growth,
                                                                 total_time_steps);
    
    // Output data for plotting
    std::string output_filename = "output.csv";
    std::vector<std::vector<double>> all_vectors = {effective_temperatures};
    write_vectors_to_csv(output_filename, all_vectors);

    return 0;
}
