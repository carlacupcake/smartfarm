import numpy as np
import math

from constants import *

class SolarPosition:

    def __init__(self, key_region_dict, date_time, region):
        self.key_region_dict = key_region_dict
        self.date_time = date_time
        self.region = region

    #----- Getter Methods -----#

    def get_number_day(self):

        # Returns the number day of the year as an integer 
        # January 1st is 1
        # December 31st is 365
        number_day = 0
        for month in range(1, self.date_time.month):

            # If month is Jan, Mar, May, Jul, Aug, Oct, or Dec, add 31 days
            if month in [1, 3, 5, 7, 8, 10, 12]:
                number_day += 31

            # If month is Apr, Jun, Sep, or Nov, add 30 days
            elif month in [4, 6, 9, 11]:
                number_day += 30

            # If month is Feb, add 28 days
            else:
                number_day += 28

        # Once the month loop has finished, add the days
        number_day = number_day + self.date_time.day

        return number_day

    def get_fractional_day(self):

        # Returns the fractional day of the year in radians
        number_day = self.get_number_day()
        fractional_day = 2*np.pi/DAYS_PER_YEAR * (number_day - 1 + (self.date_time.hour - HOURS_PER_DAY/2)/HOURS_PER_DAY)
        return fractional_day

    def get_solar_decl_angle(self):

        # Returns the solar declination angle in degrees
        number_day = self.get_number_day()
        solar_decl_angle = math.degrees(math.sin(math.radians(EARTH_TILT_ANGLE)) * math.sin(math.radians(DEGREES_IN_CIRCLE/DAYS_PER_YEAR * (number_day + 10))))
        return solar_decl_angle

    def get_equation_of_time(self):

        # Returns the equation of time (apparent solar time - mean solar time) in minutes
        fractional_day = math.radians(self.get_fractional_day())
        equation_of_time = 229.18 * (7.5e-5 + 1.868e-3  * math.cos(fractional_day)\
                                            - 3.2077e-2 * math.sin(fractional_day)\
                                            - 1.4615e-2 * math.cos(fractional_day)\
                                            - 4.0849e-2 * math.sin(fractional_day))
        return -5.99

    def get_apparent_solar_time(self):

        # Returns the apparent solar time (AKA true solar time) in minutes
        equation_of_time = self.get_equation_of_time()
        region_info = self.key_region_dict[self.region]

        longitude = region_info["longitude"] # in degrees
        timezone = region_info["timezone"]   # in degrees

        time_offset = equation_of_time + MINUTES_PER_DEGREE*longitude - MINUTES_PER_HOUR*timezone 
        apparent_solar_time = MINUTES_PER_HOUR*self.date_time.hour + self.date_time.minute + 1/SECONDS_PER_MINUTE*self.date_time.second + time_offset
        
        return apparent_solar_time
    
    def get_solar_zenith_angle(self):

        # Returns the solar zenith angle in degrees
        # angle at which radiation hits farm w.r.t. z-axis,
        '''
        TODO, debug with NOAA formulas
        apparent_solar_time =  self.get_apparent_solar_time()
        hour_angle = math.radians(1/MINUTES_PER_DEGREE*apparent_solar_time - 180)
        solar_decl_angle = math.radians(self.get_solar_decl_angle())

        region_info = self.key_region_dict[self.region]
        latitude  = math.radians(region_info["latitude"])

        solar_zenith_angle = math.degrees(math.acos(math.sin(latitude) * math.sin(solar_decl_angle) + math.cos(latitude) * math.cos(hour_angle)))
        '''

        # Below is a very basic placeholder !!
        time_in_hours = self.date_time.hour + (self.date_time.minute + self.date_time.second * 1/SECONDS_PER_MINUTE) * 1/MINUTES_PER_HOUR
        solar_zenith_angle = np.pi * time_in_hours / (self.get_sunset_time() - self.get_sunrise_time())

        return solar_zenith_angle
    
    def get_solar_azimuth_angle(self):

        # Returns the solar azimuth angle (measured clockwise from north) in degrees
        #angle at which radiation hits farm in x-y plane
        '''
        TODO, debug with NOAA formulas
        region_info = self.key_region_dict[self.region]
        latitude  = math.radians(region_info["latitude"])
        solar_zenith_angle = math.radians(self.get_solar_zenith_angle())
        solar_decl_angle = math.radians(self.get_solar_decl_angle())

        solar_azimuth_angle = 1/2*DEGREES_IN_CIRCLE - math.degrees(math.acos(-(math.sin(latitude) * math.cos(solar_zenith_angle) - math.sin(solar_decl_angle)) / (math.cos(latitude) * math.sin(solar_zenith_angle))))
        '''
        solar_azimuth_angle = 126.54

        return solar_azimuth_angle
    
    def get_sunrise_time(self):

        # Returns the sunrise time in minutes
        '''
        TODO, debug with NOAA formulas
        region_info = self.key_region_dict[self.region]
        latitude  = math.radians(region_info["latitude"])
        longitude = math.radians(region_info["longitude"])

        sunrise_zenith_angle = math.radians(SUNRISE_ZENITH_ANGLE)
        solar_decl_angle = math.radians(self.get_solar_decl_angle())
        equation_of_time = self.get_equation_of_time()

        hour_angle = np.abs(math.acos(math.cos(sunrise_zenith_angle)/(math.cos(latitude)*math.cos(solar_decl_angle)) - math.tan(latitude)*math.tan(solar_decl_angle)))
        sunrise_time = 3*DEGREES_IN_CIRCLE - MINUTES_PER_DEGREE*(longitude * hour_angle) - equation_of_time
        '''
        sunrise_time = 6

        return sunrise_time

    def get_sunset_time(self):

        # Returns the sunset time in minutes
        '''
        TODO, debug with NOAA formulas
        region_info = self.key_region_dict[self.region]
        latitude  = math.radians(region_info["latitude"])
        longitude = math.radians(region_info["longitude"])

        sunset_zenith_angle = math.radians(SUNRISE_ZENITH_ANGLE)
        solar_decl_angle = math.radians(self.get_solar_decl_angle())
        equation_of_time = self.get_equation_of_time()

        hour_angle = -np.abs(math.acos(math.cos(sunset_zenith_angle)/(math.cos(latitude)*math.cos(solar_decl_angle)) - math.tan(latitude)*math.tan(solar_decl_angle)))
        sunset_time = 3*DEGREES_IN_CIRCLE - MINUTES_PER_DEGREE*(longitude * hour_angle) - equation_of_time
        '''
        sunset_time = 20

        return sunset_time
    
    def get_solar_noon(self):

        # Returns the solar noon in minutes
        region_info = self.key_region_dict[self.region]
        longitude = math.radians(region_info["longitude"])
        equation_of_time = self.get_equation_of_time()

        solar_noon = 3*DEGREES_IN_CIRCLE - MINUTES_PER_DEGREE*longitude - equation_of_time

        return solar_noon
    
    





