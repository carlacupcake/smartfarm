// simulation.cpp
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <locale>
#include <ctime> // For std::tm

#include "constants.h"
#include "datetime.h"
#include "user_input.h"
#include "utilities.h"

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
    int radiation_column_index   = 0;
    int temperature_column_index = 1;
    bool has_header              = false;

    std::vector<double> hourly_radiation = load_data_from_csv_column(disturbance_data_filename, 
                                                                     radiation_column_index, 
                                                                     has_header);
    std::vector<double> hourly_temperature = load_data_from_csv_column(disturbance_data_filename, 
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

    // Calculate average temperature over simulation length
    std::vector<double> average_temperatures = calc_avg_daily_values(hourly_temperature,
                                                                     total_time_steps);
    std::vector<double> average_irrigation = calc_avg_daily_values(hourly_irrigation,
                                                                   total_time_steps);

    // Calculate leaf and fruit sensitivity values over the simulation length
    std::vector<double> leaf_sensitivity_temp = calc_leaf_sens_values(average_temperatures,
                                                                       temp_critical_fruit_growth,
                                                                       temp_optimal_fruit_growth,
                                                                       total_time_steps);
    std::vector<double> fruit_sensitivity_temp = calc_fruit_sens_values(average_temperatures,
                                                                        leaf_sensitivity_temp,
                                                                        temp_critical_fruit_growth,
                                                                        temp_ceiling_fruit_growth,
                                                                        temp_optimal_fruit_growth,
                                                                        total_time_steps);
    std::vector<double> leaf_sensitivity_water = calc_leaf_sens_values(average_irrigation,
                                                                       water_critical_fruit_growth,
                                                                       water_optimal_fruit_growth,
                                                                       total_time_steps);

    // Calculate effective values over simulation length
    std::vector<double> effective_temperatures = calc_eff_values(hourly_temperature,
                                                                 temp_critical_fruit_growth,
                                                                 total_time_steps);
    std::vector<double> effective_irrigation = calc_eff_values(hourly_irrigation,
                                                               water_critical_fruit_growth,
                                                               total_time_steps);
    std::vector<double> effective_fertilizer = calc_eff_values(hourly_fertilizer,
                                                               fertilizer_critical_fruit_growth,
                                                               total_time_steps);

    // Calculate cumulative values over simulation length
    std::vector<double> cumulative_temperatures = calc_cum_values(effective_temperatures,
                                                                  total_time_steps);
    std::vector<double> cumulative_irrigation = calc_cum_values(effective_irrigation,
                                                                total_time_steps);
    std::vector<double> cumulative_fertilizer = calc_cum_values(effective_fertilizer,
                                                                total_time_steps);
    
    // Save input data to CSV for plotting
    std::string output_filename = "farm_sim_inputs.csv";
    std::vector<std::vector<double>> all_eff_vectors = {hourly_temperature,
                                                        hourly_irrigation,
                                                        hourly_fertilizer,
                                                        effective_temperatures,
                                                        effective_irrigation,
                                                        effective_fertilizer,
                                                        cumulative_temperatures,
                                                        cumulative_irrigation,
                                                        cumulative_fertilizer,
                                                        leaf_sensitivity_temp,
                                                        fruit_sensitivity_temp,
                                                        leaf_sensitivity_water};
    write_vectors_to_csv(output_filename, all_eff_vectors);

    //--- MAIN SIMULATION ---//

    // Set triggers
    bool height_decay_triggered = false;
    bool leaf_decay_triggered = false;
    bool fruit_canopy_decay_triggered = false;

    // Initialize storage for state variables
    std::vector<double> height(total_time_steps, initial_height);
    std::vector<double> leaf_area(total_time_steps, initial_leaf_area);
    std::vector<double> canopy_biomass(total_time_steps, initial_canopy_biomass);
    std::vector<double> fruit_biomass(total_time_steps, initial_fruit_biomass);

    // Begin simulation
    int days_after_sowing = 0;
    DateTime todays_date = start_date;
    for (int day = 0; day < total_simulation_days; ++day) {

        // Print out the day
        std::cout << "Day " << (day + 1) << " of " << total_simulation_days << std::endl;

        // Update days after sowing and today's date
        days_after_sowing += 1;
        todays_date.day += 1;

        // Loop over time steps (for updates that happen throughout a single day)
        for (int time_step = 0; time_step < num_time_steps_per_day; ++time_step) {

            // Index for the time-dependent variables
            int t = day * num_time_steps_per_day + time_step;
            if (t == (total_time_steps + 1)) {
                break; // cannot fill the t+1 position if this condition is true, so break the loop
            }

            // Update today's date to the hour
            todays_date.hour += 1;

            // TODO: Add SolarPosition logic
            double time_in_hours = todays_date.hour + (todays_date.minute + todays_date.second * 1/SECONDS_PER_MINUTE) * 1/MINUTES_PER_HOUR;
            double solar_zenith_angle = M_PI * time_in_hours / (sunset_hour - sunrise_hour);

            // Plant Height Update
            double dhdt = 0;
            double logistic_growth_term = growth_rate_height * height[t] * (1 - height[t]/carrying_capacity_height);
            double decay_term = -1 * decay_rate_height * height[t];
            double gaussian_irrigation_term = 0;
            double gaussian_fertilizer_term = 0;

            if (effective_irrigation[t] == 0) {
                gaussian_irrigation_term = 0;
            }
            else {
                gaussian_irrigation_term = gains_height_from_water * growth_rate_height_water_specific * effective_irrigation[t] \
                                           * std::exp(-std::pow((cumulative_irrigation[t] - scaling_factor_peak_height_growth_water_specific * effective_irrigation[t]) / \
                                                                (scaling_factor_peak_growth_time_height_water_specific * effective_irrigation[t]), 2));
            }
            if (effective_fertilizer[t] == 0) {
                gaussian_fertilizer_term = 0;
            }
            else {
                gaussian_fertilizer_term = gains_height_from_fertilizer * growth_rate_height_fertilizer_specific * effective_fertilizer[t] \
                                           * std::exp(-std::pow((cumulative_fertilizer[t] - scaling_factor_peak_height_growth_fertilizer_specific * effective_fertilizer[t]) / \
                                                                (scaling_factor_peak_growth_time_height_fertilizer_specific * effective_fertilizer[t]), 2));
            }

            if (height[t] > carrying_capacity_height) {
                height_decay_triggered = true;
            }
            if (height_decay_triggered) {
                dhdt = logistic_growth_term + gaussian_irrigation_term + gaussian_fertilizer_term + decay_term;
            }
            else {
                dhdt = logistic_growth_term + gaussian_irrigation_term + gaussian_fertilizer_term;
            }
            height[t+1] = height[t] + dhdt * time_step_size;

            // Leaf area update 
            double dAdt = 0;
            double temperature_decay_term = -1 * decay_rate_leaf_area_temp_specific * leaf_sensitivity_temp[day] * leaf_area[t];
            double irrigation_decay_term = -1 * decay_rate_leaf_area_water_specific * leaf_sensitivity_water[day] * leaf_area[t];
            double gaussian_temperature_term = 0;
            gaussian_irrigation_term = 0;
            gaussian_fertilizer_term = 0;

            if (effective_temperatures[t] == 0) {
                gaussian_temperature_term = 0;
            }
            else { 
                gaussian_temperature_term = gains_leaf_area_from_temp * growth_rate_leaf_area_temp_specific * effective_temperatures[t] \
                                            * std::exp(-std::pow((cumulative_temperatures[t] - scaling_factor_peak_leaf_area_growth_temp_specific * effective_temperatures[t]) / \
                                                                 (scaling_factor_peak_growth_time_leaf_area_temp_specific * effective_temperatures[t]), 2));
            }
            if (effective_irrigation[day] == 0) {
                gaussian_irrigation_term = 0;
            }
            else {
                gaussian_irrigation_term = gains_leaf_area_from_water * growth_rate_leaf_area_water_specific * effective_irrigation[t] \
                                           * std::exp(-std::pow((cumulative_irrigation[t] - scaling_factor_peak_leaf_area_growth_water_specific * effective_irrigation[day]) / \
                                                                (scaling_factor_peak_growth_time_leaf_area_water_specific * effective_irrigation[t]), 2));
            }
            if (effective_fertilizer[day] == 0) {
                gaussian_fertilizer_term = 0;
            }
            else {
                gaussian_fertilizer_term = gains_leaf_area_from_fertilizer * growth_rate_leaf_area_fertilizer_specific * effective_fertilizer[t] \
                                          * std::exp(-std::pow((cumulative_fertilizer[t] - scaling_factor_peak_leaf_area_growth_fertilizer_specific * effective_fertilizer[day]) / \
                                                               (scaling_factor_peak_growth_time_leaf_area_fertilizer_specific * effective_fertilizer[t]), 2));
            }
            
            if (leaf_area[t] > carrying_capacity_leaf_area) {
                leaf_decay_triggered = true;
            }
            if (height[t] > threshold_height_leaf_growth) { // Seedling --> vegetative transition
                if (leaf_decay_triggered) {            
                    dAdt = gaussian_temperature_term + gaussian_irrigation_term + gaussian_fertilizer_term + temperature_decay_term + irrigation_decay_term;
                }                
                else {
                    dAdt = gaussian_temperature_term + gaussian_irrigation_term + gaussian_fertilizer_term;
                }
            }
            else {
                dAdt = 0;
            }
            leaf_area[t+1] = leaf_area[t] + dAdt * time_step_size;

            // For each of the plants, determine how well light passes through the canopy
            // First, circumscribe squares about the circles that represents the canopies
            double radius = std::sqrt(leaf_area[t] / M_PI);
            double square_side_lengths = 2 * radius;

            // Get the coordinates that define the center of the lower left plant (our representative plants)
            double xcoord_canopy = 0; // TODO change to actual coords
            double ycoord_canopy = 0; // TODO change to actual coords

            // Get the coordinates that define the lower left corners of the squares
            double start_x_square = xcoord_canopy - radius;
            double start_y_square = ycoord_canopy - radius;

            // Discretize the canopy of each plant - find how to discretize in x and y
            double side_length_leaf_test_point = std::sqrt(area_leaf_test_point);
            int num_test_points_x_per_plant = static_cast<int>(std::floor(square_side_lengths / side_length_leaf_test_point));
            int num_test_points_y_per_plant = num_test_points_x_per_plant;

            // Calculate the canopy density for each plant
            double canopy_density = 0;
            if (leaf_area[t] != 0) {
                canopy_density = 3 * canopy_biomass[t] / leaf_area[t] * std::sqrt(M_PI/ leaf_area[t]) * std::tan(conal_angle_leaves_wrt_stem);
            } else {
                canopy_density = 0;
            }

            double conal_angle_leaves_wrt_origin = conal_angle_leaves_wrt_stem + field_zenith_angle;

            // For the representative plant, tile the square that circumscribes the canopy
            double initial_rad = hourly_radiation[t];
            double avg_irradiance_leaf_surface = initial_rad; 
            for (int j = 0; j < num_test_points_x_per_plant; ++j) {

                double x = start_x_square + j * side_length_leaf_test_point;
                for (int k = 0; k < num_test_points_y_per_plant; ++k) {

                    double y = start_y_square + k * side_length_leaf_test_point;

                    // Only evaluate irradiance at test point if it is within the canopy
                    double r = std::sqrt(std::pow(x, 2) + std::pow(y, 2));
                    if (r <= radius) {

                        // Calculate the azimuthal angle of the point
                        double azimuth_angle_test_point_wrt_stem = std::tan(y/x); // radians
                        double azimuth_angle_test_point_wrt_origin = azimuth_angle_test_point_wrt_stem + field_azimuth_angle;
                        
                        // Calculated just in case, but unused for now
                        //radius_wrt_stem = np.sqrt(x**2 + y**2)
                        //z_wrt_height = radius_wrt_stem * 1/math.tan(conal_angle_leaves_wrt_stem)

                        // Calculate the extinction coefficient for this test_point
                        double extinction_angle_photosynthesis = conal_angle_leaves_wrt_origin - solar_zenith_angle * std::cos(azimuth_angle_test_point_wrt_origin - solar_azimuth_angle);
                        double extinction_coefficient_photosynthesis = std::sin(extinction_angle_photosynthesis);

                        // Calculate the irradiance at this test point
                        double irradiance_test_point = initial_rad * (1 - std::exp(-extinction_coefficient_photosynthesis * canopy_density/typical_canopy_density));

                        // Update the average irradiance across this canopy
                        avg_irradiance_leaf_surface = (avg_irradiance_leaf_surface + irradiance_test_point)/2;

                    }
                }
            }

            // Update the total irradiance at the leaf surface for the representative plant     
            double irradiance_leaf_surface = avg_irradiance_leaf_surface;

        }

    }

    return 0;
}
