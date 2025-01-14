// constants.h
#ifndef CONSTANTS_H
#define CONSTANTS_H

// Constants for date, time, earth, and sun
extern const int    DAYS_PER_YEAR;
extern const int    HOURS_PER_DAY;
extern const double DEGREES_IN_CIRCLE;
extern const double MINUTES_PER_DEGREE;
extern const double MINUTES_PER_HOUR;
extern const double SECONDS_PER_MINUTE;
extern const double SUNRISE_ZENITH_ANGLE;
extern const double EARTH_TILT_ANGLE;
extern const double MONTHS_PER_YEAR;
extern const int DEFAULT_START_MINUTE;
extern const int DEFAULT_START_SECOND;

// TODO - TEMPORARY CONSTANTS -> Use SolarPosition Logic to Calculate
extern double sunrise_hour;
extern double sunset_hour;
extern double solar_azimuth_angle;

// Variable Glossary - Field Configuration
extern const int num_plants_x;
extern const int num_plants_y;
extern const int num_plants;
extern const double field_dim_x;
extern const double field_dim_y;
extern const double field_azimuth_angle;
extern const double field_zenith_angle;

// Variable Glossary - Simulation Settings
extern const double time_step_size;
extern const double area_leaf_test_point;
extern const double density_leaf_test_points;
extern const int num_time_steps_per_day;

// Variable Glossary - Inital Conditions for Time-Dependent Variables
extern const int days_after_sowing;
extern const double initial_height;
extern const double initial_leaf_area;
extern const double initial_irradiance_leaf_surface;
extern const double initial_canopy_biomass;
extern const double initial_fruit_biomass;

// Variable Glossary - Growth Rates
extern const double growth_rate_canopy_biomass;
extern const double growth_rate_fruit_biomass;
extern const double growth_rate_height;
extern const double growth_rate_height_fertilizer_specific;
extern const double growth_rate_height_water_specific;
extern const double growth_rate_leaf_area_fertilizer_specific;
extern const double growth_rate_leaf_area_temp_specific;
extern const double growth_rate_leaf_area_water_specific;

// Variable Glossary - Peak Growth Times
extern const double scaling_factor_peak_growth_time_height_fertilizer_specific;
extern const double scaling_factor_peak_growth_time_height_water_specific;
extern const double scaling_factor_peak_growth_time_leaf_area_fertilizer_specific;
extern const double scaling_factor_peak_growth_time_leaf_area_temp_specific;
extern const double scaling_factor_peak_growth_time_leaf_area_water_specific;

// Variable Glossary - Other Peak Growth Parameters
extern const double scaling_factor_peak_height_growth_fertilizer_specific;
extern const double scaling_factor_peak_height_growth_water_specific;
extern const double scaling_factor_peak_leaf_area_growth_fertilizer_specific;
extern const double scaling_factor_peak_leaf_area_growth_temp_specific;
extern const double scaling_factor_peak_leaf_area_growth_water_specific;

// Variable Glossary - Decay Rates
extern const double decay_rate_canopy_biomass;
extern const double decay_rate_fruit_biomass;
extern const double decay_rate_height;
extern const double decay_rate_leaf_area_temp_specific;
extern const double decay_rate_leaf_area_water_specific;

// Variable Glossary - Carrying Capacities
extern const double carrying_capacity_canopy_biomass;
extern const double carrying_capacity_fruit_biomass;
extern const double carrying_capacity_height;
extern const double carrying_capacity_leaf_area;

// Variable Glossary - Gains from Various Stimuli
extern const double gains_height_from_fertilizer;
extern const double gains_height_from_water;
extern const double gains_leaf_area_from_fertilizer;
extern const double gains_leaf_area_from_temp;
extern const double gains_leaf_area_from_water;

// Variable Glossary - Thresholds for Growth/Decay
extern const double threshold_canopy_fruit_growth;
extern const double threshold_height_leaf_growth;
extern const double threshold_dAdt_canopy_fruit_decay;

// Variable Glossary - Other Crop-specific Parameters
extern const double absorption_factor_single_leaf;
extern const double conal_angle_leaves_wrt_stem;
extern const double efficiency_coefficient_photosynthesis;
extern const double fruit_biomass_temp_sensitivity_parameter;
extern const double typical_canopy_density;

// Variable Glossary - Critical, Ceiling, and Optimal Values
extern const double fertilizer_critical_fruit_growth;
extern const double temp_ceiling_fruit_growth;
extern const double temp_critical_fruit_growth;
extern const double temp_optimal_fruit_growth;
extern const double water_ceiling_fruit_growth;
extern const double water_critical_fruit_growth;
extern const double water_optimal_fruit_growth;

#endif // CONSTANTS_H