/**
 * \file integrator_godunov_first_order.c
 * 
 * \brief First-order Godunov scheme for the Euler equations.
 * 
 * \author Ching-Yin Ng
 * \date 2025-03-19
 */

#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef USE_OPENMP
#include <omp.h>
#endif

#include "hydro.h"
#include "progress_bar.h"
#include "reconstruction.h"
#include "riemann_solver.h"
#include "source_term.h"
#include "utils.h"


ErrorStatus godunov_first_order_1d(
    System *__restrict system,
    IntegratorParam *__restrict integrator_param,
    StoringParam *__restrict storing_param,
    Settings *__restrict settings,
    SimulationParam *__restrict simulation_param,
    SimulationStatus *__restrict simulation_status
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

    /* Arrays */
    double *__restrict mass = system->mass_;
    double *__restrict momentum_x = system->momentum_x_;
    double *__restrict energy = system->energy_;
    double *__restrict surface_area_x = system->surface_area_x_;

    double *__restrict interface_density_L = system->interface_density_x_L_;
    double *__restrict interface_density_R = system->interface_density_x_R_;
    double *__restrict interface_velocity_x_L = system->interface_velocity_x_x_L_;
    double *__restrict interface_velocity_x_R = system->interface_velocity_x_x_R_;
    double *__restrict interface_pressure_L = system->interface_pressure_x_L_;
    double *__restrict interface_pressure_R = system->interface_pressure_x_R_;

    /* System parameters */
    const double gamma = system->gamma;
    const int num_cells = system->num_cells_x;
    const int num_ghost_cells_side = system->num_ghost_cells_side;
    const int num_interfaces = num_cells + 1;

    /* Integrator parameters */
    const double cfl = integrator_param->cfl;
    const double cfl_initial_shrink_factor = integrator_param->cfl_initial_shrink_factor;
    const int cfl_initial_shrink_num_steps = integrator_param->cfl_initial_shrink_num_steps;

    /* Storing parameters */
    const bool is_storing = storing_param->is_storing;
    const double storing_interval = storing_param->storing_interval;
    int *__restrict store_count_ptr = &storing_param->store_count_;

    /* Settings */
    const bool no_progress_bar = settings->no_progress_bar;

    /* Simulation parameters */
    const double tf = simulation_param->tf;

    /* Simulation status */
    int64 *__restrict num_steps_ptr = &simulation_status->num_steps;
    double *__restrict dt_ptr = &simulation_status->dt;
    double *__restrict t_ptr = &simulation_status->t;

    /* Reconstruction */

    /* Main Loop */
    ProgressBarParam progress_bar_param;
    if (!no_progress_bar)
    {
        start_progress_bar(&progress_bar_param, tf);
    }

    *num_steps_ptr = 0;
    *dt_ptr = 0.0;
    *t_ptr = 0.0;

    /* Store first snapshot */
    *store_count_ptr = 0;
    const int store_initial_offset = (is_storing && storing_param->store_initial) ? 1 : 0;
    if (is_storing && storing_param->store_initial)
    {
        error_status = WRAP_TRACEBACK(store_snapshot(system, integrator_param, simulation_status, storing_param));
        if (error_status.return_code != SUCCESS)
        {
            goto err_store_first_snapshot;
        }
    }

    while (*t_ptr < tf)
    {
        /* Compute time step */
        if (*num_steps_ptr < cfl_initial_shrink_num_steps)
        {
            *dt_ptr = get_time_step_1d(system, cfl * cfl_initial_shrink_factor);
        }
        else
        {
            *dt_ptr = get_time_step_1d(system, cfl);
        }
        if (*t_ptr + (*dt_ptr) > tf)
        {
            *dt_ptr = tf - *t_ptr;
        }
        if ((*dt_ptr) == 0.0)
        {
            error_status = WRAP_RAISE_ERROR(VALUE_ERROR, "dt is zero.");
            goto err_dt_zero;
        }
        const double dt = (*dt_ptr);

        /* Compute fluxes and update step */
        error_status = reconstruct_cell_interface_1d(system, integrator_param);
        if (error_status.return_code != SUCCESS)
        {
            goto err_reconstruct_cell_interface;
        }
#ifdef USE_OPENMP
        #pragma omp parallel for
#endif
        for (int i = 0; i < num_interfaces; i++)
        {
            double flux_mass;
            double flux_momentum_x;
            double flux_energy;
            ErrorStatus local_error_status = WRAP_TRACEBACK(solve_flux_1d(
                integrator_param,
                settings,
                &flux_mass,
                &flux_momentum_x,
                &flux_energy,
                gamma,
                interface_density_L[i],
                interface_velocity_x_L[i],
                interface_pressure_L[i],
                interface_density_R[i],
                interface_velocity_x_R[i],
                interface_pressure_R[i]
            ));
            if (local_error_status.return_code != SUCCESS)
            {
                error_status = local_error_status;
            }

            const double d_rho = flux_mass * dt;
            const double d_rho_u = flux_momentum_x * dt;
            const double d_energy_density = flux_energy * dt;

            const int idx = num_ghost_cells_side + i;
            mass[idx - 1] -= d_rho * surface_area_x[idx - 1];
            momentum_x[idx - 1] -= d_rho_u * surface_area_x[idx - 1];
            energy[idx - 1] -= d_energy_density * surface_area_x[idx - 1];

            mass[idx] += d_rho * surface_area_x[idx];
            momentum_x[idx] += d_rho_u * surface_area_x[idx];
            energy[idx] += d_energy_density * surface_area_x[idx];
        }

        if (error_status.return_code != SUCCESS)
        {
            goto err_solve_flux;
        }

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

        (*t_ptr) += dt;
        (*num_steps_ptr)++;

        /* Store snapshot */
        if (is_storing && ((*t_ptr) >= storing_interval * (*store_count_ptr - store_initial_offset + 1)))
        {
            error_status = WRAP_TRACEBACK(store_snapshot(system, integrator_param, simulation_status, storing_param));
            if (error_status.return_code != SUCCESS)
            {
                goto err_store_snapshot;
            }
        }

        if (!no_progress_bar)
        {
            update_progress_bar(&progress_bar_param, *t_ptr, false);
        }
    }

    if (!no_progress_bar)
    {
        update_progress_bar(&progress_bar_param, *t_ptr, true);
    }

    return make_success_error_status();

err_store_snapshot:
err_set_boundary_condition:
err_convert_conserved_to_primitive:
err_compute_geometry_source_term:
err_solve_flux:
err_reconstruct_cell_interface:
err_dt_zero:
    if (!no_progress_bar)
    {
        update_progress_bar(&progress_bar_param, *t_ptr, true);
    }
err_store_first_snapshot:
err_coord_sys:
    return error_status;
}

ErrorStatus godunov_first_order_2d(
    System *__restrict system,
    IntegratorParam *__restrict integrator_param,
    StoringParam *__restrict storing_param,
    Settings *__restrict settings,
    SimulationParam *__restrict simulation_param,
    SimulationStatus *__restrict simulation_status
)
{
    /* Declare variables */
    ErrorStatus error_status;

    /* Check coordinate system */
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

    /* Arrays */
    double *__restrict mass = system->mass_;
    double *__restrict momentum_x = system->momentum_x_;
    double *__restrict momentum_y = system->momentum_y_;
    double *__restrict energy = system->energy_;
    double *__restrict surface_area_x = system->surface_area_x_;
    double *__restrict surface_area_y = system->surface_area_y_;

    double *__restrict interface_density_x_L = system->interface_density_x_L_;
    double *__restrict interface_density_x_R = system->interface_density_x_R_;
    double *__restrict interface_density_y_B = system->interface_density_y_B_;
    double *__restrict interface_density_y_T = system->interface_density_y_T_;
    double *__restrict interface_velocity_x_x_L = system->interface_velocity_x_x_L_;
    double *__restrict interface_velocity_x_x_R = system->interface_velocity_x_x_R_;
    double *__restrict interface_velocity_x_y_L = system->interface_velocity_x_y_L_;
    double *__restrict interface_velocity_x_y_R = system->interface_velocity_x_y_R_;
    double *__restrict interface_velocity_y_x_B = system->interface_velocity_y_x_B_;
    double *__restrict interface_velocity_y_x_T = system->interface_velocity_y_x_T_;
    double *__restrict interface_velocity_y_y_B = system->interface_velocity_y_y_B_;
    double *__restrict interface_velocity_y_y_T = system->interface_velocity_y_y_T_;
    double *__restrict interface_pressure_x_L = system->interface_pressure_x_L_;
    double *__restrict interface_pressure_x_R = system->interface_pressure_x_R_;
    double *__restrict interface_pressure_y_B = system->interface_pressure_y_B_;
    double *__restrict interface_pressure_y_T = system->interface_pressure_y_T_;

    /* System parameters */
    const double gamma = system->gamma;
    const int num_ghost_cells_side = system->num_ghost_cells_side;
    const int num_cells_x = system->num_cells_x;
    const int num_cells_y = system->num_cells_y;
    const int total_num_cells_x = num_cells_x + 2 * num_ghost_cells_side;
    const int num_interfaces_x = num_cells_x + 1;
    const int num_interfaces_y = num_cells_y + 1;

    /* Integrator parameters */
    const double cfl = integrator_param->cfl;
    const double cfl_initial_shrink_factor = integrator_param->cfl_initial_shrink_factor;
    const int cfl_initial_shrink_num_steps = integrator_param->cfl_initial_shrink_num_steps;

    /* Storing parameters */
    const bool is_storing = storing_param->is_storing;
    const double storing_interval = storing_param->storing_interval;
    int *__restrict store_count_ptr = &storing_param->store_count_;

    /* Settings */
    const bool no_progress_bar = settings->no_progress_bar;

    /* Simulation parameters */
    const double tf = simulation_param->tf;

    /* Simulation status */
    int64 *__restrict num_steps_ptr = &simulation_status->num_steps;
    double *__restrict dt_ptr = &simulation_status->dt;
    double *__restrict t_ptr = &simulation_status->t;

    /* Main Loop */
    ProgressBarParam progress_bar_param;
    if (!no_progress_bar)
    {
        start_progress_bar(&progress_bar_param, tf);
    }

    *num_steps_ptr = 0;
    *dt_ptr = 0.0;
    *t_ptr = 0.0;

    /* Store first snapshot */
    *store_count_ptr = 0;
    const int store_initial_offset = (is_storing && storing_param->store_initial) ? 1 : 0;
    if (is_storing && storing_param->store_initial)
    {
        error_status = WRAP_TRACEBACK(store_snapshot(system, integrator_param, simulation_status, storing_param));
        if (error_status.return_code != SUCCESS)
        {
            goto err_store_first_snapshot;
        }
    }

    while (*t_ptr < tf)
    {
        /* Compute time step */
        if (*num_steps_ptr < cfl_initial_shrink_num_steps)
        {
            *dt_ptr = get_time_step_2d(system, cfl * cfl_initial_shrink_factor);
        }
        else
        {
            *dt_ptr = get_time_step_2d(system, cfl);
        }
        if (*t_ptr + (*dt_ptr) > tf)
        {
            *dt_ptr = tf - *t_ptr;
        }
        if ((*dt_ptr) == 0.0)
        {
            error_status = WRAP_RAISE_ERROR(VALUE_ERROR, "dt is zero.");
            goto err_dt_zero;
        }
        const double dt = (*dt_ptr);

        /* Compute fluxes and update step */
        error_status = reconstruct_cell_interface_2d(system, integrator_param);
        if (error_status.return_code != SUCCESS)
        {
            goto err_reconstruct_cell_interface;
        }
#ifdef USE_OPENMP
        #pragma omp parallel for
#endif
        for (int i = 0; i < num_interfaces_x; i++)
        {
            for (int j = 0; j < num_interfaces_y; j++)
            {
                ErrorStatus local_error_status;

                /* x-direction */
                double flux_mass_x;
                double flux_momentum_x_x;
                double flux_momentum_x_y;
                double flux_energy_x;

                const int idx_interface_x = j * num_interfaces_x + i;
                local_error_status = WRAP_TRACEBACK(solve_flux_2d(
                    integrator_param,
                    settings,
                    &flux_mass_x,
                    &flux_momentum_x_x,
                    &flux_momentum_x_y,
                    &flux_energy_x,
                    gamma,
                    interface_density_x_L[idx_interface_x],
                    interface_velocity_x_x_L[idx_interface_x],
                    interface_velocity_x_y_L[idx_interface_x],
                    interface_pressure_x_L[idx_interface_x],
                    interface_density_x_R[idx_interface_x],
                    interface_velocity_x_x_R[idx_interface_x],
                    interface_velocity_x_y_R[idx_interface_x],
                    interface_pressure_x_R[idx_interface_x]
                ));
                if (local_error_status.return_code != SUCCESS)
                {
                    error_status = local_error_status;
                }

                /* y-direction */
                double flux_mass_y;
                double flux_momentum_y_x;
                double flux_momentum_y_y;
                double flux_energy_y;
                const int idx_interface_y = j * num_interfaces_x + i;
                local_error_status = WRAP_TRACEBACK(solve_flux_2d(
                    integrator_param,
                    settings,
                    &flux_mass_y,
                    &flux_momentum_y_y,
                    &flux_momentum_y_x,
                    &flux_energy_y,
                    gamma,
                    interface_density_y_B[idx_interface_y],
                    interface_velocity_y_y_B[idx_interface_y],
                    interface_velocity_y_x_B[idx_interface_y],
                    interface_pressure_y_B[idx_interface_y],
                    interface_density_y_T[idx_interface_y],
                    interface_velocity_y_y_T[idx_interface_y],
                    interface_velocity_y_x_T[idx_interface_y],
                    interface_pressure_y_T[idx_interface_y]
                ));
                if (local_error_status.return_code != SUCCESS)
                {
                    error_status = local_error_status;
                }

                const int idx_i = num_ghost_cells_side + i;
                const int idx_j = num_ghost_cells_side + j;
                mass[idx_j * total_num_cells_x + (idx_i - 1)] -= dt * flux_mass_x * surface_area_x[idx_i - 1];
                momentum_x[idx_j * total_num_cells_x + (idx_i - 1)] -= dt * flux_momentum_x_x * surface_area_x[idx_i - 1];
                momentum_y[idx_j * total_num_cells_x + (idx_i - 1)] -= dt * flux_momentum_x_y * surface_area_x[idx_i - 1];
                energy[idx_j * total_num_cells_x + (idx_i - 1)] -= dt * flux_energy_x * surface_area_x[idx_i - 1];

                mass[(idx_j - 1) * total_num_cells_x + idx_i] -= dt * flux_mass_y * surface_area_y[idx_j - 1];
                momentum_x[(idx_j - 1) * total_num_cells_x + idx_i] -= dt * flux_momentum_y_x * surface_area_y[idx_j - 1];
                momentum_y[(idx_j - 1) * total_num_cells_x + idx_i] -= dt * flux_momentum_y_y * surface_area_y[idx_j - 1];
                energy[(idx_j - 1) * total_num_cells_x + idx_i] -= dt * flux_energy_y * surface_area_y[idx_j - 1];

                mass[idx_j * total_num_cells_x + idx_i] += dt * (flux_mass_x * surface_area_x[idx_i] + flux_mass_y * surface_area_y[idx_j]);
                momentum_x[idx_j * total_num_cells_x + idx_i] += dt * (flux_momentum_x_x * surface_area_x[idx_i] + flux_momentum_y_x * surface_area_y[idx_j]);
                momentum_y[idx_j * total_num_cells_x + idx_i] += dt * (flux_momentum_x_y * surface_area_x[idx_i] + flux_momentum_y_y * surface_area_y[idx_j]);
                energy[idx_j * total_num_cells_x + idx_i] += dt * (flux_energy_x * surface_area_x[idx_i] + flux_energy_y * surface_area_y[idx_j]);
            }
        }

        if (error_status.return_code != SUCCESS)
        {
            goto err_solve_flux;
        }

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

        (*t_ptr) += dt;
        (*num_steps_ptr)++;

        /* Store snapshot */
        if (is_storing && ((*t_ptr) >= storing_interval * (*store_count_ptr - store_initial_offset + 1)))
        {
            error_status = WRAP_TRACEBACK(store_snapshot(system, integrator_param, simulation_status, storing_param));
            if (error_status.return_code != SUCCESS)
            {
                goto err_store_snapshot;
            }
        }

        if (!no_progress_bar)
        {
            update_progress_bar(&progress_bar_param, *t_ptr, false);
        }
    }

    if (!no_progress_bar)
    {
        update_progress_bar(&progress_bar_param, *t_ptr, true);
    }

    return make_success_error_status();

err_store_snapshot:
err_set_boundary_condition:
err_convert_conserved_to_primitive:
err_solve_flux:
err_reconstruct_cell_interface:
err_dt_zero:
    if (!no_progress_bar)
    {
        update_progress_bar(&progress_bar_param, *t_ptr, true);
    }
err_store_first_snapshot:
err_coord_sys:
    return error_status;
}
