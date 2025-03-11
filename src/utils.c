/**
 * \file utils.c
 * 
 * \brief Utility functions for the hydrodynamics simulation.
 * 
 * \author Ching-Yin Ng
 * \date 2025-03-11
 */

#include <math.h>

#include "hydro.h"

double get_sound_speed(const double gamma, const double rho, const double p)
{
    return sqrt(gamma * p / rho);
}

double get_time_step_1d(
    const System *__restrict system,
    const double cfl
)
{
    const double gamma = system->gamma;
    const double *__restrict density = system->density_;
    const double *__restrict pressure = system->pressure_;
    const double *__restrict velocity = system->velocity_x_;
    const int num_cells = system->num_cells_x;
    const int num_ghost_cells_side = system->num_ghost_cells_side;

    double S_max = -INFINITY;
    
    for (int i = num_ghost_cells_side; i < (num_cells + num_ghost_cells_side); i++)
    {
        const double a = get_sound_speed(gamma, density[i], pressure[i]);
        const double v = fabs(velocity[i]);
        if (a + v > S_max)
        {
            S_max = a + v;
        }
    }

    return cfl * system->dx_ / S_max;
}

double get_time_step_2d(
    const System *__restrict system,
    const double cfl
)
{
    const double gamma = system->gamma;
    const double *__restrict density = system->density_;
    const double *__restrict pressure = system->pressure_;
    const double *__restrict velocity_x = system->velocity_x_;
    const double *__restrict velocity_y = system->velocity_y_;
    const int num_cells_x = system->num_cells_x;
    const int num_cells_y = system->num_cells_y;
    const int num_ghost_cells_side = system->num_ghost_cells_side;
    const int total_num_cells_x = num_cells_x + 2 * num_ghost_cells_side;
    const double dx = system->dx_;
    const double dy = system->dy_;

    double S_max_x = -INFINITY;
    double S_max_y = -INFINITY;

    for (int i = num_ghost_cells_side; i < (num_cells_x + num_ghost_cells_side); i++)
    {
        for (int j = num_ghost_cells_side; j < (num_cells_y + num_ghost_cells_side); j++)
        {
            const int idx = j * total_num_cells_x + i;
            const double a = get_sound_speed(gamma, density[idx], pressure[idx]);
            const double v_x = fabs(velocity_x[idx]);
            const double v_y = fabs(velocity_y[idx]);
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

    const double factor = fmin(dx / S_max_x, dy / S_max_y);

    return cfl * factor;
}
