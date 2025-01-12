// datetime.h
#ifndef DATETIME_H
#define DATETIME_H

#include <ctime>  // For std::tm, std::time_t

struct DateTime {
    int year;
    int month;
    int day;
    int hour;
    int minute;
    int second;

    // Constructor
    DateTime(int y, int m, int d, int h, int min, int sec)
        : year(y), month(m), day(d), hour(h), minute(min), second(sec) {}

    // Convert to std::tm
    std::tm toTm() const {
        std::tm date_time = {};
        date_time.tm_year = year - 1900; // tm_year is years since 1900
        date_time.tm_mon = month - 1;    // tm_mon is 0-indexed
        date_time.tm_mday = day;
        date_time.tm_hour = hour;
        date_time.tm_min = minute;
        date_time.tm_sec = second;
        return date_time;
    }

    // Convert to epoch time
    std::time_t toEpochTime() const {
        std::tm date_time = toTm();
        return std::mktime(&date_time);
    }

    // Get the equivalent NSRDB point
    int get_nsrdb_equiv_point() const;

    // Get equivalent day of simulation
    int get_number_day() const;
};

#endif // DATETIME_H
