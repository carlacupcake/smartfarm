// user_input.h
#ifndef USER_INPUT_H
#define USER_INPUT_H
#include <string>

// Genetic Algorithm User Inputs
const double water_period      = 24;  // plants are watered every x hours    [hours, in {0, 2927}]
const double fertilizer_period = 720; // plants are fertilized every x hours [hours, in {0, 2927}]

const double water_amount_per_plant      = 0.2; // amount of water delivered to each plant each time it is watered         [kg] 
const double fertilizer_amount_per_plant = 0.2; // amount of fertilizer delivered to each plant each time it is fertilized [kg] 

// Non-Genetic Algorithm User Inputs
const int start_year   = 2024; // year in which the simulation starts   [unitless, in {1, 2024}]
const int start_month  = 5;    // month in which the simulation starts  [unitless, in {1, 12}]
const int start_day    = 31;   // day on which the simulation starts    [unitless, in {1, 31}]
const int start_hour   = 12;   // hour in which the simulation starts   [unitless, in {1, 24}]
const int start_minute = 0;    // minute in which the simulation starts [unitless, in {1, 60}]
const int start_second = 0;    // second at which the simulation starts [unitless, in {1, 60}]

const int end_year     = 2024; // year in which the simulation ends     [unitless, in {1, 2024}]
const int end_month    = 9;    // month in which the simulation ends    [unitless, in {1, 12}]
const int end_day      = 30;   // day on which the simulation ends      [unitless, in {1, 31}]
const int end_hour     = 12;   // hour in which the simulation ends     [unitless, in {1, 24}]
const int end_minute   = 0;    // minute in which the simulation ends   [unitless, in {1, 60}]
const int end_second   = 0;    // second at which the simulation ends   [unitless, in {1, 60}]

const int water_start_year   = start_year;           // year in which the first watering occurs   [unitless, in {1, 2024}]
const int water_start_month  = 5;                    // month in which the first watering occurs  [unitless, in {1, 12}]
const int water_start_day    = 31;                   // day on which the first watering occurs    [unitless, in {1, 31}]
const int water_start_hour   = 9;                    // hour in which the first watering occurs   [unitless, in {1, 24}]
const int water_start_minute = DEFAULT_START_MINUTE; // minute in which the first watering occurs [unitless, in {1, 60}]
const int water_start_second = DEFAULT_START_SECOND; // second at which the first watering occurs [unitless, in {1, 60}]

const int fertilizer_start_year   = start_year;            // year in which the first fertilization occurs   [unitless, in {1, 2024}]
const int fertilizer_start_month  = 5;                     // month in which the first fertilization occurs  [unitless, in {1, 12}]
const int fertilizer_start_day    = 31;                    // day on which the first fertilization occurs    [unitless, in {1, 31}]
const int fertilizer_start_hour   = 15;                    // hour in which the first fertilization occurs   [unitless, in {1, 24}]
const int fertilizer_start_minute = DEFAULT_START_MINUTE;  // minute in which the first fertilization occurs [unitless, in {1, 60}]
const int fertilizer_start_second = DEFAULT_START_SECOND;  // second at which the first fertilization occurs [unitless, in {1, 60}]

#endif // USER_INPUT_H