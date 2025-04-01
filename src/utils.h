/**
 * \file utils.h
 * 
 * \brief Utility functions for the hydrodynamics simulation.
 * 
 * \author Ching-Yin Ng
 * \date April 2025
 */

#ifndef UTILS_H
#define UTILS_H

#include "hydro.h"


/**
 * \brief Get current time as a decimal number of seconds using clock_gettime(CLOCK_MONOTONIC, )
 * 
 * \return Current time as a decimal number of seconds
 */
double hydro_get_current_time(void);

/**
 * \brief Compute the sound speed corresponding to the given density and pressure.
 * 
 * \param gamma Adiabatic index.
 * \param rho Density.
 * \param p Pressure.
 * 
 * \retval Sound speed a = sqrt(gamma * p / rho).
 */
double get_sound_speed(double gamma, double rho, double p);

/**
 * \brief Get the time step for 1D system based on the CFL condition.
 * 
 * Calculate dt = cfl * dx / S_max, where S_max = max{|u| + a}. Note
 * that this can lead to an underestimate of S_max. For instance, 
 * if u = 0 at t0, then S_max = a_max, which results in dt thats
 * too large. It is advised to use a much smaller cfl for the initial
 * steps until the flow has developed.
 * 
 * \param system System object.
 * \param cfl CFL number.
 * 
 * \retval Time step dt.
 */
double get_time_step_1d(
    const System *__restrict system,
    const double cfl
);

/**
 * \brief Get the time step for 2D system based on the CFL condition.
 * 
 * Calculate dt = cfl * min{dx / S_max_x, dy / S_max_y}, where 
 * S_max_x = max{|u| + a} and S_max_y = max{|v| + a}. Note
 * that this can lead to an underestimate of S_max. It is advised to 
 * use a much smaller cfl for the initial steps until the flow has developed.
 * 
 * \param system System object.
 * \param cfl CFL number.
 * 
 * \retval Time step dt.
 */
double get_time_step_2d(
    const System *__restrict system,
    const double cfl
);

#endif
