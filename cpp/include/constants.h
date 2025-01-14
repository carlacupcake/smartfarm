// constants.h
#ifndef CONSTANTS_H
#define CONSTANTS_H
#include <cmath>  // For std::ceil

// Constants for date, time, earth, and sun
const int    DAYS_PER_YEAR        = 365;
const int    HOURS_PER_DAY        = 24;
const double DEGREES_IN_CIRCLE    = 360;
const double MINUTES_PER_DEGREE   = 4;
const double MINUTES_PER_HOUR     = 60;
const double SECONDS_PER_MINUTE   = 60;
const double SUNRISE_ZENITH_ANGLE = 90.833;
const double EARTH_TILT_ANGLE     = 23.5;
const double MONTHS_PER_YEAR      = 12;
const int DEFAULT_START_MINUTE    = 0;
const int DEFAULT_START_SECOND    = 0;

// Variable Glossary - Field Configuration
const int num_plants_x           = 17;                          // number of plants in a row in the x-direction                [unitless]
const int num_plants_y           = 9;                           // number of plants in a column in the y-direction             [unitless]
const int num_plants             = num_plants_x * num_plants_y; // number of plants in the field,                              [unitless]
const double field_dim_x         = 16;                          // length of the field in the x-direction                      [m]
const double field_dim_y         = 8;                           // width of the field in the y-direction                       [m]
const double field_azimuth_angle = 0;                           // clockwise rotation of farm origin from north,               [radians]
const double field_zenith_angle  = 0;                           // surface normal of plane in which field lives w.r.t. z-axis, [radians]  

// Variable Glossary - Simulation Settings
const double time_step_size           = 1.0;                                                         // simulation time step size in hours,    [hours] 
const double area_leaf_test_point     = 0.005;                                                       // area of leaf test point,               [m2]
const double density_leaf_test_points = 1/area_leaf_test_point;                                      // density of test points within a leaf,  [points/m2]
const int num_time_steps_per_day      = static_cast<int>(std::ceil(HOURS_PER_DAY / time_step_size)); // number of time steps in one day,       [unitless]

// Variable Glossary - Inital Conditions for Time-Dependent Variables
const int days_after_sowing                  = 0;      // days after sowing,                                          [days]
const double initial_height                  = 0.01;   // initial height of representative plant,                     [m]
const double initial_leaf_area               = 1.0e-5; // initial leaf area of representative plant,                  [m2]
const double initial_irradiance_leaf_surface = 0.01;   // initial irradiance at leaf surface of representative plant, [W/m2]
const double initial_canopy_biomass          = 0.01;   // initial canopy biomass of representative plant,             [kg]
const double initial_fruit_biomass           = 0.01;   // initial fruit biomass of representative plant,              [kg]

// Variable Glossary - Growth Rates
const double growth_rate_canopy_biomass                = 1.0e0;  // 1/s
const double growth_rate_fruit_biomass                 = 1.0e0;  // 1/s
const double growth_rate_height                        = 8.0e-3; // 1/s
const double growth_rate_height_fertilizer_specific    = 7.5e-2; // 1/s
const double growth_rate_height_water_specific         = 7.5e-2; // 1/s
const double growth_rate_leaf_area_fertilizer_specific = 8.0e-3; // 1/s
const double growth_rate_leaf_area_temp_specific       = 8.0e-2; // 1/s
const double growth_rate_leaf_area_water_specific      = 8.0e-3; // 1/s

// Variable Glossary - Peak Growth Times
const double scaling_factor_peak_growth_time_height_fertilizer_specific    = 3.5e1; // 1/s
const double scaling_factor_peak_growth_time_height_water_specific         = 3.5e1; // 1/s
const double scaling_factor_peak_growth_time_leaf_area_fertilizer_specific = 2.0e1; // 1/s
const double scaling_factor_peak_growth_time_leaf_area_temp_specific       = 2.0e1; // 1/s
const double scaling_factor_peak_growth_time_leaf_area_water_specific      = 2.0e1; // 1/s

// Variable Glossary - Other Peak Growth Parameters
const double scaling_factor_peak_height_growth_fertilizer_specific    = 8.0e1; // unitless
const double scaling_factor_peak_height_growth_water_specific         = 8.0e1; // unitless
const double scaling_factor_peak_leaf_area_growth_fertilizer_specific = 6.5e1; // unitless
const double scaling_factor_peak_leaf_area_growth_temp_specific       = 6.5e1; // unitless
const double scaling_factor_peak_leaf_area_growth_water_specific      = 3.7e1; // unitless

// Variable Glossary - Decay Rates
const double decay_rate_canopy_biomass           = 1.0e-2; // 1/s
const double decay_rate_fruit_biomass            = 1.0e-2; // 1/s
const double decay_rate_height                   = 5.0e-3; // 1/s
const double decay_rate_leaf_area_temp_specific  = 1.0e-3; // 1/s/degC
const double decay_rate_leaf_area_water_specific = 8.0e-4; // 1/kg

// Variable Glossary - Carrying Capacities
const double carrying_capacity_canopy_biomass = 1.0e0;   // m2
const double carrying_capacity_fruit_biomass  = 5.0e0;   // kg
const double carrying_capacity_height         = 1.0e0;   // m
const double carrying_capacity_leaf_area      = 15.0e-4; // m2

// Variable Glossary - Gains from Various Stimuli
const double gains_height_from_fertilizer    = 4.0e-5; // m/kg
const double gains_height_from_water         = 4.0e-5; // m/kg
const double gains_leaf_area_from_fertilizer = 6.0e-6; // m2/kg
const double gains_leaf_area_from_temp       = 2.5-6;  // m2/degC
const double gains_leaf_area_from_water      = 6.0e-6; // m2/kg

// Variable Glossary - Thresholds for Growth/Decay
const double threshold_canopy_fruit_growth     = 2.5e-1;  // kg
const double threshold_height_leaf_growth      = 2.5e-1;  // m
const double threshold_dAdt_canopy_fruit_decay = -2.5e-6; // m2/s

// Variable Glossary - Other Crop-specific Parameters
const double absorption_factor_single_leaf            = 1.0e0;  // unitless
const double conal_angle_leaves_wrt_stem              = 0.785;  // radians (45 degrees)
const double efficiency_coefficient_photosynthesis    = 5.0e-2; // 1/W
const double fruit_biomass_temp_sensitivity_parameter = 1.0e0;  // unitless
const double typical_canopy_density                   = 1.0e0;  // kg/m3

// Variable Glossary - Critical, Ceiling, and Optimal Values
const double fertilizer_critical_fruit_growth = 0.0e0;  // kg
const double temp_ceiling_fruit_growth        = 3.7e1;  // degC
const double temp_critical_fruit_growth       = 0.0e0;  // degC
const double temp_optimal_fruit_growth        = 2.3e1;  // degC
const double water_ceiling_fruit_growth       = 1.0e3;  // kg
const double water_critical_fruit_growth      = 0.0e1;  // kg
const double water_optimal_fruit_growth       = 1.0e-2; // kg

#endif // CONSTANTS_H