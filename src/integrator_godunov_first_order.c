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
#include <string.h>

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
        size_t error_message_size = strlen(
            "Wrong coordinate system. Supported coordinate system: \"cartesian_1d\", \"cylindrical_1d\" and \"spherical_1d\", got: \"\""
        ) + strlen(system->coord_sys) + 1;
        char *__restrict error_message = malloc(error_message_size * sizeof(char));
        if (!error_message)
        {
            error_status = WRAP_RAISE_ERROR(MEMORY_ERROR, "Memory allocation for error message failed.");
            goto err_coord_sys;
        }
        snprintf(
            error_message,
            error_message_size,
            "Wrong coordinate system. Supported coordinate system: \"cartesian_1d\", \"cylindrical_1d\" and \"spherical_1d\", got: \"%s\"",
            system->coord_sys
        );
        error_status = WRAP_RAISE_ERROR(VALUE_ERROR, error_message);
        goto err_coord_sys;
    }

    bool is_compute_geometry_source_term = true;
    if (system->coord_sys_flag_ == COORD_SYS_CARTESIAN_1D)
    {
        is_compute_geometry_source_term = false;
    }

    const real gamma = system->gamma;
    real *__restrict mass = system->mass_;
    real *__restrict momentum_x = system->momentum_x_;
    real *__restrict energy = system->energy_;
    real *__restrict surface_area_x = system->surface_area_x_;
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

        /* Compute fluxes and update step */
        for (int i = num_ghost_cells_side; i < (num_ghost_cells_side + num_interfaces); i++)
        {
            real flux_mass;
            real flux_momentum_x;
            real flux_energy;
            error_status = WRAP_TRACEBACK(solve_flux_1d(
                integrator_param,
                settings,
                &flux_mass,
                &flux_momentum_x,
                &flux_energy,
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
        
            const real d_rho = flux_mass * dt;
            const real d_rho_u = flux_momentum_x * dt;
            const real d_energy_density = flux_energy * dt;

            mass[i - 1] -= d_rho * surface_area_x[i - 1];
            momentum_x[i - 1] -= d_rho_u * surface_area_x[i - 1];
            energy[i - 1] -= d_energy_density * surface_area_x[i - 1];

            mass[i] += d_rho * surface_area_x[i];
            momentum_x[i] += d_rho_u * surface_area_x[i];
            energy[i] += d_energy_density * surface_area_x[i];
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

    return make_success_error_status();

err_set_boundary_condition:
err_convert_conserved_to_primitive:
err_compute_geometry_source_term:
err_solve_flux:
err_dt_zero:
    if (!no_progress_bar)
    {
        update_progress_bar(&progress_bar_param, t, true);
    }
err_coord_sys:
    return error_status;
}

WIN32DLL_API ErrorStatus godunov_first_order_2d(
    System *__restrict system,
    IntegratorParam *__restrict integrator_param,
    StoringParam *__restrict storing_param,
    Settings *__restrict settings,
    SimulationParam *__restrict simulation_param
)
{
    /* Declare variables */
    ErrorStatus error_status;

    if (system->coord_sys_flag_ != COORD_SYS_CARTESIAN_2D)
    {
        size_t error_message_size = strlen(
            "Wrong coordinate system. Supported coordinate system: \"cartesian_2d\", got: \"\""
        ) + strlen(system->coord_sys) + 1;
        char *__restrict error_message = malloc(error_message_size * sizeof(char));
        if (!error_message)
        {
            error_status = WRAP_RAISE_ERROR(MEMORY_ERROR, "Memory allocation for error message failed.");
            goto err_coord_sys;
        }
        snprintf(
            error_message,
            error_message_size,
            "Wrong coordinate system. Supported coordinate system: \"cartesian_1d\", got: \"%s\"",
            system->coord_sys
        );
        error_status = WRAP_RAISE_ERROR(VALUE_ERROR, error_message);
        goto err_coord_sys;
    }

    const real gamma = system->gamma;
    real *__restrict mass = system->mass_;
    real *__restrict momentum_x = system->momentum_x_;
    real *__restrict momentum_y = system->momentum_y_;
    real *__restrict energy = system->energy_;
    real *__restrict surface_area_x = system->surface_area_x_;
    real *__restrict surface_area_y = system->surface_area_y_;
    real *__restrict volume = system->volume_;
    real *__restrict density = system->density_;
    real *__restrict velocity_x = system->velocity_x_;
    real *__restrict velocity_y = system->velocity_y_;
    real *__restrict pressure = system->pressure_;
    const int num_ghost_cells_side = system->num_ghost_cells_side;
    const int num_cells_x = system->num_cells_x;
    const int num_cells_y = system->num_cells_y;
    const int total_num_cells_x = num_cells_x + 2 * num_ghost_cells_side;
    const int total_num_cells_y = num_cells_y + 2 * num_ghost_cells_side;

    const real cfl = integrator_param->cfl;
    const real cfl_initial_shrink_factor = integrator_param->cfl_initial_shrink_factor;
    const int num_steps_shrink = integrator_param->num_steps_shrink;

    const bool no_progress_bar = settings->no_progress_bar;

    const real tf = simulation_param->tf;

    real t = 0.0;

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
            dt = get_time_step_2d(system, cfl * cfl_initial_shrink_factor);
        }
        else
        {
            dt = get_time_step_2d(system, cfl);
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

        /* Compute fluxes and update step */
        for (int i = num_ghost_cells_side; i < (2 * num_ghost_cells_side + num_cells_x); i++)
        {
            for (int j = num_ghost_cells_side; j < (2 * num_ghost_cells_side + num_cells_y); j++)
            {
                /* x-direction */
                real flux_mass_x;
                real flux_momentum_x_x;
                real flux_momentum_x_y;
                real flux_energy_x;

                error_status = WRAP_TRACEBACK(solve_flux_2d(
                    integrator_param,
                    settings,
                    &flux_mass_x,
                    &flux_momentum_x_x,
                    &flux_momentum_x_y,
                    &flux_energy_x,
                    gamma,
                    density[j * total_num_cells_x + (i - 1)],
                    velocity_x[j * total_num_cells_x + (i - 1)],
                    velocity_y[j * total_num_cells_x + (i - 1)],
                    pressure[j * total_num_cells_x + (i - 1)],
                    density[j * total_num_cells_x + i],
                    velocity_x[j * total_num_cells_x + i],
                    velocity_y[j * total_num_cells_x + i],
                    pressure[j * total_num_cells_x + i]
                ));
                if (error_status.return_code != SUCCESS)
                {
                    goto err_solve_flux;
                }

                /* y-direction */
                real flux_mass_y;
                real flux_momentum_y_x;
                real flux_momentum_y_y;
                real flux_energy_y;
                error_status = WRAP_TRACEBACK(solve_flux_2d(
                    integrator_param,
                    settings,
                    &flux_mass_y,
                    &flux_momentum_y_y,
                    &flux_momentum_y_x,
                    &flux_energy_y,
                    gamma,
                    density[(j - 1) * total_num_cells_x + i],
                    velocity_y[(j - 1) * total_num_cells_x + i],
                    velocity_x[(j - 1) * total_num_cells_x + i],
                    pressure[(j - 1) * total_num_cells_x + i],
                    density[j * total_num_cells_x + i],
                    velocity_y[j * total_num_cells_x + i],
                    velocity_x[j * total_num_cells_x + i],
                    pressure[j * total_num_cells_x + i]
                ));
                if (error_status.return_code != SUCCESS)
                {
                    goto err_solve_flux;
                }

                mass[j * total_num_cells_x + (i - 1)] -= dt * flux_mass_x * surface_area_x[i - 1];
                momentum_x[j * total_num_cells_x + (i - 1)] -= dt * flux_momentum_x_x * surface_area_x[i - 1];
                momentum_y[j * total_num_cells_x + (i - 1)] -= dt * flux_momentum_x_y * surface_area_x[i - 1];
                energy[j * total_num_cells_x + (i - 1)] -= dt * flux_energy_x * surface_area_x[i - 1];

                mass[(j - 1) * total_num_cells_x + i] -= dt * flux_mass_y * surface_area_y[j - 1];
                momentum_x[(j - 1) * total_num_cells_x + i] -= dt * flux_momentum_y_x * surface_area_y[j - 1];
                momentum_y[(j - 1) * total_num_cells_x + i] -= dt * flux_momentum_y_y * surface_area_y[j - 1];
                energy[(j - 1) * total_num_cells_x + i] -= dt * flux_energy_y * surface_area_y[j - 1];

                mass[j * total_num_cells_x + i] += dt * (flux_mass_x * surface_area_x[i] + flux_mass_y * surface_area_y[j]);
                momentum_x[j * total_num_cells_x + i] += dt * (flux_momentum_x_x * surface_area_x[i] + flux_momentum_y_x * surface_area_y[j]);
                momentum_y[j * total_num_cells_x + i] += dt * (flux_momentum_x_y * surface_area_x[i] + flux_momentum_y_y * surface_area_y[j]);
                energy[j * total_num_cells_x + i] += dt * (flux_energy_x * surface_area_x[i] + flux_energy_y * surface_area_y[j]);
            }
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

        if (!no_progress_bar)
        {
            update_progress_bar(&progress_bar_param, t, false);
        }
    }

    if (!no_progress_bar)
    {
        update_progress_bar(&progress_bar_param, t, true);
    }

    return make_success_error_status();


err_solve_flux:
err_dt_zero:
    if (!no_progress_bar)
    {
        update_progress_bar(&progress_bar_param, t, true);
    }
err_set_boundary_condition:
err_convert_conserved_to_primitive:
err_coord_sys:
    return error_status;
}
