/**
 * \file integrator_muscl_hancock.c
 * 
 * \brief Definitions of the MUSCL-Hancock scheme for Euler's equations.
 * 
 * \author Ching-Yin Ng
 * \date April 2025
 */

#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef USE_OPENMP
#include <omp.h>
#endif

#include "hydro.h"
#include "progress_bar.h"
#include "riemann_solver.h"
#include "source_term.h"
#include "utils.h"

ErrorStatus muscl_hancock_1d(
    BoundaryConditionParam *__restrict boundary_condition_param,
    System *__restrict system,
    IntegratorParam *__restrict integrator_param,
    StoringParam *__restrict storing_param,
    Settings *__restrict settings,
    SimulationParam *__restrict simulation_param,
    SimulationStatus *__restrict simulation_status
)
{
    /* Declare variables */
    ErrorStatus error_status = make_success_error_status();

    bool is_compute_geometry_source_term = true;
    if (system->coord_sys_flag_ == COORD_SYS_CARTESIAN_1D)
    {
        is_compute_geometry_source_term = false;
    }

    /* System parameters */
    const double gamma = system->gamma;
    const int num_cells = system->num_cells_x;
    const int num_ghost_cells_side = system->num_ghost_cells_side;
    const int num_interfaces = num_cells + 1;
    const double dx = system->dx_;

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

    /* Arrays */
    double *__restrict density = system->density_;
    double *__restrict velocity_x = system->velocity_x_;
    double *__restrict pressure = system->pressure_;
    double *__restrict mass = system->mass_;
    double *__restrict momentum_x = system->momentum_x_;
    double *__restrict energy = system->energy_;
    double *__restrict surface_area_x = system->surface_area_x_;
    double *__restrict volume = system->volume_;

    double *__restrict interface_density_L = malloc(num_interfaces * sizeof(double));
    double *__restrict interface_density_R = malloc(num_interfaces * sizeof(double));
    double *__restrict interface_velocity_x_L = malloc(num_interfaces * sizeof(double));
    double *__restrict interface_velocity_x_R = malloc(num_interfaces * sizeof(double));
    double *__restrict interface_pressure_L = malloc(num_interfaces * sizeof(double));
    double *__restrict interface_pressure_R = malloc(num_interfaces * sizeof(double));
    if (
        !interface_density_L
        || !interface_density_R
        || !interface_velocity_x_L
        || !interface_velocity_x_R
        || !interface_pressure_L
        || !interface_pressure_R
    )
    {
        error_status = WRAP_RAISE_ERROR(MEMORY_ERROR, "Failed to allocate memory for interface arrays.");
        goto err_memory_interface;
    }

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
        error_status = WRAP_TRACEBACK(store_snapshot(boundary_condition_param, system, integrator_param, simulation_status, storing_param));
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

        /* Reconstruct interface */
        for (int i = 0; i < (num_cells + 2); i++)
        {
            const int idx_i = num_ghost_cells_side + i - 1;

            const double density_i = density[idx_i];
            const double velocity_x_i = velocity_x[idx_i];
            const double pressure_i = pressure[idx_i];

            const double sound_speed = get_sound_speed(gamma, density_i, pressure_i);

            // Calculate the field difference
            const double delta_density_L = density_i - density[idx_i - 1];
            const double delta_velocity_x_L = velocity_x_i - velocity_x[idx_i - 1];
            const double delta_pressure_L = pressure_i - pressure[idx_i - 1];

            const double delta_density_R = density[idx_i + 1] - density[idx_i];
            const double delta_velocity_x_R = velocity_x[idx_i + 1] - velocity_x[idx_i];
            const double delta_pressure_R = pressure[idx_i + 1] - pressure[idx_i];

            double delta_density;
            double delta_velocity_x;
            double delta_pressure;

            /* Slope limiting */
            limit_slope(
                &delta_density,
                integrator_param,
                delta_density_L,
                delta_density_R
            );
            limit_slope(
                &delta_velocity_x,
                integrator_param,
                delta_velocity_x_L,
                delta_velocity_x_R
            );
            limit_slope(
                &delta_pressure,
                integrator_param,
                delta_pressure_L,
                delta_pressure_R
            );

            /**
             *  Coefficient matrix                      
             *          / u     rho       0    \
             *   A(W) = | 0      u     1 / rho |
             *          \ 0  rho a^2      u    /
             *   
             *   \overline{W}^L_i = W_i - 1/2 [I + dt / dx A(W_i)] \Delta_i                       
             *   \overline{W}^R_i = W_i + 1/2 [I - dt / dx A(W_i)] \Delta_i
             */
            const double update_density_L = -0.5 * (
                (1.0 + (dt / dx) * velocity_x_i) * delta_density
                + (dt / dx) * density_i * delta_velocity_x
                + 0.0
            );
            const double update_velocity_x_L = -0.5 * (
                0.0 +
                (1.0 + (dt / dx) * velocity_x_i )* delta_velocity_x
                + (dt / dx) * delta_pressure / density_i
            );
            const double update_pressure_L = -0.5 * (
                0.0 +
                (dt / dx) * density_i * sound_speed * sound_speed * delta_velocity_x
                + (1.0 + (dt / dx) * velocity_x_i) * delta_pressure
            );

            const double update_density_R = 0.5 * (
                (1.0 - (dt / dx) * velocity_x_i )* delta_density
                - (dt / dx) * density_i * delta_velocity_x
                + 0.0
            );
            const double update_velocity_x_R = 0.5 * (
                0.0 +
                (1.0 - (dt / dx) * velocity_x_i) * delta_velocity_x
                - (dt / dx) * delta_pressure / density_i
            );
            const double update_pressure_R = 0.5 * (
                0.0 +
                - (dt / dx) * density_i * sound_speed * sound_speed * delta_velocity_x
                + (1.0 - (dt / dx) * velocity_x_i) * delta_pressure
            );
            
            if (i > 0)
            {
                interface_density_R[i - 1] = density_i + update_density_L;
                interface_velocity_x_R[i - 1] = velocity_x_i + update_velocity_x_L;
                interface_pressure_R[i - 1] = pressure_i + update_pressure_L;
            }
            if (i < num_cells + 1)
            {
                interface_density_L[i] = density_i + update_density_R;
                interface_velocity_x_L[i] = velocity_x_i + update_velocity_x_R;
                interface_pressure_L[i] = pressure_i + update_pressure_R;
            }
        }

    /* Compute fluxes and update step */
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
#ifdef USE_OPENMP
                #pragma omp critical
                {
#endif
                    error_status = local_error_status;
#ifdef USE_OPENMP
                }
#endif
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

        convert_conserved_to_primitive_1d(
            num_cells,
            num_ghost_cells_side,
            gamma,
            volume,
            mass,
            momentum_x,
            energy,
            density,
            velocity_x,
            pressure
        );

        error_status = set_boundary_condition_1d(
            boundary_condition_param,
            density,
            velocity_x,
            pressure,
            num_ghost_cells_side,
            num_cells
        );
        if (error_status.return_code != SUCCESS)
        {
            goto err_set_boundary_condition;
        }

        if (is_compute_geometry_source_term)
        {
            error_status = WRAP_TRACEBACK(add_geometry_source_term(boundary_condition_param, system, dt));
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
            error_status = WRAP_TRACEBACK(store_snapshot(boundary_condition_param, system, integrator_param, simulation_status, storing_param));
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

    free(interface_density_L);
    free(interface_density_R);
    free(interface_velocity_x_L);
    free(interface_velocity_x_R);
    free(interface_pressure_L);
    free(interface_pressure_R);

    return make_success_error_status();

err_store_snapshot:
err_compute_geometry_source_term:
err_set_boundary_condition:
err_dt_zero:
    if (!no_progress_bar)
    {
        update_progress_bar(&progress_bar_param, *t_ptr, true);
    }
err_store_first_snapshot:
err_memory_interface:
    free(interface_density_L);
    free(interface_density_R);
    free(interface_velocity_x_L);
    free(interface_velocity_x_R);
    free(interface_pressure_L);
    free(interface_pressure_R);

    return error_status;
}
