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

#endif
