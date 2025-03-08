/**
 * \file utils.h
 * 
 * \brief Header file for utility functions for the hydrodynamics simulation.
 * 
 * \author Ching-Yin Ng
 * \date 2025-03-08
 */

#ifndef UTILS_H
#define UTILS_H

#include "hydro.h"

/**
 * \brief Compute the sound speed corresponding to the given density and pressure.
 * 
 * \param gamma Adiabatic index.
 * \param rho Density.
 * \param p Pressure.
 * 
 * \retval Sound speed a = sqrt(gamma * p / rho).
 */
real get_sound_speed(real gamma, real rho, real p);

/**
 * \brief Get the time step for 1D system based on the CFL condition.
 * 
 * Calculate dt = cfl * dx / S_max, where S_max = max{|u| + a}. Note
 * that this can lead to an underestimate of S_max. For instance, at
 * t = 0, if u = 0, then S_max = a_max, which results in dt thats
 * too large. It is advised to use a much smaller cfl for the initial
 * steps until the flow has developed.
 * 
 * \param system System object.
 * \param cfl CFL number.
 * 
 * \retval Time step dt.
 */
real get_time_step_1d(
    const System *__restrict system,
    const real cfl
);

#endif
