// datetime.cpp
#include "constants.h" 
#include "datetime.h"

int DateTime::get_nsrdb_equiv_point() const {

    int equiv_point = 0;
    
    for (int m = 1; m < month; ++m) {

        if (m == 1 || m == 3 || m == 5 || m == 7 || m == 8 || m == 10 || m == 12) {

            // If month is Jan, Mar, May, Jul, Aug, Oct, or Dec, add 31 * 24 points
            equiv_point += 31 * HOURS_PER_DAY; 

        } else if (m == 4 || m == 6 || m == 9 || m == 11) {

            // If month is Apr, Jun, Sep, or Nov, add 30 * 24 points
            equiv_point += 30 * HOURS_PER_DAY;

        } else {

            // If month is Feb, add 28 * 24 points
            equiv_point += 28 * HOURS_PER_DAY;

        }
    }

    // Once the month loop has finished, add points for the days
    equiv_point += (day - 1) * HOURS_PER_DAY;

    // Once points for the days have been added, add points for the hours
    equiv_point += hour;

    return equiv_point;
}

int DateTime::get_number_day() const {

    // Returns the number day of the year as an integer 
    // January 1st is 1
    // December 31st is 365
    int number_day = 0;

    for (int m = 1; m < month; ++m) {

        if (m == 1 || m == 3 || m == 5 || m == 7 || m == 8 || m == 10 || m == 12) {

            // If month is Jan, Mar, May, Jul, Aug, Oct, or Dec, add 31 days
            number_day += 31; 

        } else if (m == 4 || m == 6 || m == 9 || m == 11) {

            // If month is Apr, Jun, Sep, or Nov, add 30 days
            number_day += 30;

        } else {

            // If month is Feb, add 28 days
            number_day += 28;

        }
    }

    // Once the month loop has finished, add the days
    number_day = number_day + day;

    return number_day;

}
