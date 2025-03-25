/**
 * \file hydro_time.h
 * 
 * \brief Header file for getting the current time.
 * 
 * \author Ching-Yin Ng
 * \date 2025-03-25
 */

#ifndef HYDRO_TIME_H
#define HYDRO_TIME_H

/**
 * \brief Get current time as a decimal number of seconds using clock_gettime(CLOCK_MONOTONIC, )
 * 
 * \return Current time as a decimal number of seconds
 */
double hydro_get_current_time(void);

#endif
