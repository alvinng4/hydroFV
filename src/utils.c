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

WIN32DLL_API real get_sound_speed(const real gamma, const real rho, const real p)
{
    return sqrt(gamma * p / rho);
}

WIN32DLL_API real get_time_step_1d(
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

    real S_max = -INFINITY;
    
    for (int i = num_ghost_cells_side; i < (num_cells + num_ghost_cells_side); i++)
    {
        const real a = get_sound_speed(gamma, density[i], pressure[i]);
        const real v = fabs(velocity[i]);
        if (a + v > S_max)
        {
            S_max = a + v;
        }
    }

    return cfl * dx / S_max;
}

WIN32DLL_API real get_time_step_2d(
    const System *__restrict system,
    const real cfl
)
{
    const real gamma = system->gamma;
    const real *__restrict density = system->density_;
    const real *__restrict pressure = system->pressure_;
    const real *__restrict velocity_x = system->velocity_x_;
    const real *__restrict velocity_y = system->velocity_y_;
    const int num_cells_x = system->num_cells_x;
    const int num_cells_y = system->num_cells_y;
    const int num_ghost_cells_side = system->num_ghost_cells_side;
    const real dx = system->dx_;
    const real dy = system->dy_;

    real S_max_x = -INFINITY;
    real S_max_y = -INFINITY;

    for (int i = num_ghost_cells_side; i < (num_cells_x + num_ghost_cells_side); i++)
    {
        for (int j = num_ghost_cells_side; j < (num_cells_y + num_ghost_cells_side); j++)
        {
            const int idx = j * (num_cells_x + 2 * num_ghost_cells_side) + i;
            const real a = get_sound_speed(gamma, density[idx], pressure[idx]);
            const real v_x = fabs(velocity_x[idx]);
            const real v_y = fabs(velocity_y[idx]);
            if (a + v_x > S_max_x)
            {
                S_max_x = a + v_x;
            }
            if (a + v_y > S_max_y)
            {
                S_max_y = a + v_y;
            }
        }
    }

    const real factor = fmin(dx / S_max_x, dy / S_max_y);

    return cfl * factor;
}
