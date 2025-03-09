/**
 * \file integrator_godunov_first_order.c
 * 
 * \brief First-order Godunov scheme for the Euler equations.
 * 
 * \author Ching-Yin Ng
 * \date 2025-03-08
 */

#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

#include "hydro.h"
#include "progress_bar.h"
#include "riemann_solver.h"
#include "source_term.h"
#include "utils.h"


WIN32DLL_API ErrorStatus godunov_first_order_1d(
    System *__restrict system,
    IntegratorParam *__restrict integrator_param,
    StoringParam *__restrict storing_param,
    Settings *__restrict settings,
    SimulationParam *__restrict simulation_param
)
{
    /* Declare variables */
    ErrorStatus error_status;

    if (
        system->coord_sys_flag_ != COORD_SYS_CARTESIAN_1D
        && system->coord_sys_flag_ != COORD_SYS_CYLINDRICAL_1D
        && system->coord_sys_flag_ != COORD_SYS_SPHERICAL_1D
    )
    {
        error_status = WRAP_RAISE_ERROR(VALUE_ERROR, "Wrong coordinate system, expected 1D.");
        goto err_coord_sys;
    }

    bool is_compute_geometry_source_term = true;
    if (system->coord_sys_flag_ == COORD_SYS_CARTESIAN_1D)
    {
        is_compute_geometry_source_term = false;
    }

    const real gamma = system->gamma;
    real *__restrict mass = system->mass_;
    real *__restrict momentum = system->momentum_x_;
    real *__restrict energy = system->energy_;
    real *__restrict volume = system->volume_;
    real *__restrict density = system->density_;
    real *__restrict velocity_x = system->velocity_x_;
    real *__restrict pressure = system->pressure_;
    const real dx = system->dx_;
    const int num_cells = system->num_cells_x;
    const int num_ghost_cells_side = system->num_ghost_cells_side;

    const real cfl = integrator_param->cfl;
    const real cfl_initial_shrink_factor = integrator_param->cfl_initial_shrink_factor;
    const int num_steps_shrink = integrator_param->num_steps_shrink;

    const bool no_progress_bar = settings->no_progress_bar;

    const real tf = simulation_param->tf;

    const int num_interfaces = num_cells + 1;

    real t = 0.0;

    /* Allocate memory */
    real *restrict flux_mass = malloc(num_interfaces * sizeof(real));
    real *restrict flux_momentum = malloc(num_interfaces * sizeof(real));
    real *restrict flux_energy = malloc(num_interfaces * sizeof(real));
    if (!flux_mass || !flux_momentum || !flux_energy)
    {
        error_status = WRAP_RAISE_ERROR(MEMORY_ERROR, "Memory allocation failed.");
        goto err_memory;
    }

    /* Main Loop */
    ProgressBarParam progress_bar_param;
    if (!no_progress_bar)
    {
        progress_bar_param = start_progress_bar(tf);
    }

    int64 count = 0;
    while (t < tf)
    {
        /* Compute time step */
        real dt;
        if (count < num_steps_shrink)
        {
            dt = get_time_step_1d(system, cfl * cfl_initial_shrink_factor);
        }
        else
        {
            dt = get_time_step_1d(system, cfl);
        }
        if (t + dt > tf)
        {
            dt = tf - t;
        }
        if (dt == 0.0)
        {
            error_status = WRAP_RAISE_ERROR(VALUE_ERROR, "dt is zero.");
            goto err_dt_zero;
        }

        /* Compute fluxes */
        for (int i = num_ghost_cells_side; i < (num_ghost_cells_side + num_interfaces); i++)
        {
            error_status = WRAP_TRACEBACK(solve_flux(
                integrator_param,
                settings,
                &flux_mass[i - num_ghost_cells_side],
                &flux_momentum[i - num_ghost_cells_side],
                &flux_energy[i - num_ghost_cells_side],
                gamma,
                density[i - 1],
                velocity_x[i - 1],
                pressure[i - 1],
                density[i],
                velocity_x[i],
                pressure[i]
            ));
            if (error_status.return_code != SUCCESS)
            {
                goto err_solve_flux;
            }
        }

        /* Update step */
        for (int i = num_ghost_cells_side; i < (num_ghost_cells_side + num_interfaces); i++)
        {
            const real d_rho = flux_mass[i - num_ghost_cells_side] * dt / dx;
            const real d_rho_u = flux_momentum[i - num_ghost_cells_side] * dt / dx;
            const real d_energy_density = flux_energy[i - num_ghost_cells_side] * dt / dx;

            const real volume_left = volume[i - 1];
            const real volume_right = volume[i];

            mass[i - 1] -= d_rho * volume_left;
            momentum[i - 1] -= d_rho_u * volume_left;
            energy[i - 1] -= d_energy_density * volume_left;

            mass[i] += d_rho * volume_right;
            momentum[i] += d_rho_u * volume_right;
            energy[i] += d_energy_density * volume_right;
        }

        t += dt;
        count++;

        error_status = WRAP_TRACEBACK(convert_conserved_to_primitive(system));
        if (error_status.return_code != SUCCESS)
        {
            goto err_convert_conserved_to_primitive;
        }
        error_status = WRAP_TRACEBACK(set_boundary_condition(system));
        if (error_status.return_code != SUCCESS)
        {
            goto err_set_boundary_condition;
        }

        if (is_compute_geometry_source_term)
        {
            error_status = WRAP_TRACEBACK(add_geometry_source_term(system, dt));
            if (error_status.return_code != SUCCESS)
            {
                goto err_compute_geometry_source_term;
            }
        }

        if (!no_progress_bar)
        {
            update_progress_bar(&progress_bar_param, t, false);
        }
    }

    if (!no_progress_bar)
    {
        update_progress_bar(&progress_bar_param, t, true);
    }

    free(flux_mass);
    free(flux_momentum);
    free(flux_energy);

    return make_success_error_status();

err_compute_geometry_source_term:
err_set_boundary_condition:
err_convert_conserved_to_primitive:
err_dt_zero:
err_solve_flux:
err_memory:
    free(flux_mass);
    free(flux_momentum);
    free(flux_energy);
err_coord_sys:
    return error_status;
}
