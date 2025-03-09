/**
 * \file utils.c
 * 
 * \brief Utility functions for the hydrodynamics simulation.
 * 
 * \author Ching-Yin Ng
 * \date 2025-03-08
 */

#include <math.h>

#include "hydro.h"

real get_sound_speed(const real gamma, const real rho, const real p)
{
    return sqrt(gamma * p / rho);
}

real get_time_step_1d(
    const System *__restrict system,
    const real cfl
)
{
    const real gamma = system->gamma;
    const real *__restrict density = system->density_;
    const real *__restrict pressure = system->pressure_;
    const real *__restrict velocity = system->velocity_x_;
    const int num_cells = system->num_cells_x;
    const int num_ghost_cells_side = system->num_ghost_cells_side;
    const real dx = system->dx_;

    real a_max = -INFINITY;
    real v_max = -INFINITY;
    
    for (int i = num_ghost_cells_side; i < (num_cells + num_ghost_cells_side); i++)
    {
        const real a = get_sound_speed(gamma, density[i], pressure[i]);
        if (a > a_max)
        {
            a_max = a;
        }

        const real v = fabs(velocity[i]);
        if (v > v_max)
        {
            v_max = v;
        }
    }

    return cfl * dx / (a_max + v_max);
}
