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
            const double surface_area_i = surface_area_x[idx_i];
            const double volume_i = volume[idx_i];
            const double dt_A_over_vol = dt * surface_area_i / volume_i;

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
                (1.0 + dt_A_over_vol * velocity_x_i) * delta_density
                + dt_A_over_vol * density_i * delta_velocity_x
                + 0.0
            );
            const double update_velocity_x_L = -0.5 * (
                0.0
                + (1.0 + dt_A_over_vol * velocity_x_i) * delta_velocity_x
                + (dt_A_over_vol / density_i) * delta_pressure
            );
            const double update_pressure_L = -0.5 * (
                0.0
                + dt_A_over_vol * density_i * sound_speed * sound_speed * delta_velocity_x
                + (1.0 + dt_A_over_vol * velocity_x_i) * delta_pressure
            );

            const double update_density_R = 0.5 * (
                (1.0 - dt_A_over_vol * velocity_x_i )* delta_density
                - dt_A_over_vol * density_i * delta_velocity_x
                + 0.0
            );
            const double update_velocity_x_R = 0.5 * (
                0.0 +
                (1.0 - dt_A_over_vol * velocity_x_i) * delta_velocity_x
                - (dt_A_over_vol / density_i) * delta_pressure
            );
            const double update_pressure_R = 0.5 * (
                0.0 +
                - dt_A_over_vol * density_i * sound_speed * sound_speed * delta_velocity_x
                + (1.0 - dt_A_over_vol * velocity_x_i) * delta_pressure
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

ErrorStatus muscl_hancock_2d(
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
            "Wrong coordinate system. Supported coordinate system: \"cartesian_2d\", got: \"%s\"",
            system->coord_sys
        );
        error_status = WRAP_RAISE_ERROR(VALUE_ERROR, error_message);
        goto err_coord_sys;
    }

    /* System parameters */
    const double gamma = system->gamma;
    const int num_ghost_cells_side = system->num_ghost_cells_side;
    const int num_cells_x = system->num_cells_x;
    const int num_cells_y = system->num_cells_y;
    const int total_num_cells_x = num_cells_x + 2 * num_ghost_cells_side;
    const int num_interfaces_x = num_cells_x + 1;
    const int num_interfaces_y = num_cells_y + 1;

    /* Arrays */
    double *__restrict mass = system->mass_;
    double *__restrict momentum_x = system->momentum_x_;
    double *__restrict momentum_y = system->momentum_y_;
    double *__restrict energy = system->energy_;
    double *__restrict surface_area_x = system->surface_area_x_;
    double *__restrict surface_area_y = system->surface_area_y_;
    double *__restrict volume = system->volume_;
    double *__restrict density = system->density_;
    double *__restrict velocity_x = system->velocity_x_;
    double *__restrict velocity_y = system->velocity_y_;
    double *__restrict pressure = system->pressure_;

#ifndef USE_OPENMP
    /* Allocate memory for interface arrays */
    double *__restrict interface_density_x_L = malloc(num_interfaces_x * sizeof(double));
    double *__restrict interface_density_x_R = malloc(num_interfaces_x * sizeof(double));
    double *__restrict interface_velocity_x_x_L = malloc(num_interfaces_x * sizeof(double));
    double *__restrict interface_velocity_x_x_R = malloc(num_interfaces_x * sizeof(double));
    double *__restrict interface_velocity_x_y_L = malloc(num_interfaces_x * sizeof(double));
    double *__restrict interface_velocity_x_y_R = malloc(num_interfaces_x * sizeof(double));
    double *__restrict interface_pressure_x_L = malloc(num_interfaces_x * sizeof(double));
    double *__restrict interface_pressure_x_R = malloc(num_interfaces_x * sizeof(double));

    double *__restrict interface_density_y_B = malloc(num_interfaces_y * sizeof(double));
    double *__restrict interface_density_y_T = malloc(num_interfaces_y * sizeof(double));
    double *__restrict interface_velocity_y_x_B = malloc(num_interfaces_y * sizeof(double));
    double *__restrict interface_velocity_y_x_T = malloc(num_interfaces_y * sizeof(double));
    double *__restrict interface_velocity_y_y_B = malloc(num_interfaces_y * sizeof(double));
    double *__restrict interface_velocity_y_y_T = malloc(num_interfaces_y * sizeof(double));
    double *__restrict interface_pressure_y_B = malloc(num_interfaces_y * sizeof(double));
    double *__restrict interface_pressure_y_T = malloc(num_interfaces_y * sizeof(double));

    if (
        !interface_density_x_L
        || !interface_density_x_R
        || !interface_velocity_x_x_L
        || !interface_velocity_x_x_R
        || !interface_velocity_x_y_L
        || !interface_velocity_x_y_R
        || !interface_pressure_x_L
        || !interface_pressure_x_R
        || !interface_density_y_B
        || !interface_density_y_T
        || !interface_velocity_y_x_B
        || !interface_velocity_y_x_T
        || !interface_velocity_y_y_B
        || !interface_velocity_y_y_T
        || !interface_pressure_y_B
        || !interface_pressure_y_T
    )
    {
        error_status = WRAP_RAISE_ERROR(MEMORY_ERROR, "Failed to allocate memory for interface arrays.");
        goto err_memory_interface;
    }
#endif 

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

        /* Compute fluxes and update step (x-direction) */
#ifdef USE_OPENMP
        bool shared_malloc_error_flag = false;
        #pragma omp parallel
        {
            bool local_malloc_error_flag = false;
    
            /* Allocate memory for interface arrays */
            double *__restrict interface_density_x_L = malloc(num_interfaces_x * sizeof(double));
            double *__restrict interface_density_x_R = malloc(num_interfaces_x * sizeof(double));
            double *__restrict interface_velocity_x_x_L = malloc(num_interfaces_x * sizeof(double));
            double *__restrict interface_velocity_x_x_R = malloc(num_interfaces_x * sizeof(double));
            double *__restrict interface_velocity_x_y_L = malloc(num_interfaces_x * sizeof(double));
            double *__restrict interface_velocity_x_y_R = malloc(num_interfaces_x * sizeof(double));
            double *__restrict interface_pressure_x_L = malloc(num_interfaces_x * sizeof(double));
            double *__restrict interface_pressure_x_R = malloc(num_interfaces_x * sizeof(double));
    
            if (
                !interface_density_x_L
                || !interface_density_x_R
                || !interface_velocity_x_x_L
                || !interface_velocity_x_x_R
                || !interface_velocity_x_y_L
                || !interface_velocity_x_y_R
                || !interface_pressure_x_L
                || !interface_pressure_x_R
            )
            {
                local_malloc_error_flag = true;
            }
    
            // Update shared flag
            #pragma omp critical
            {
                if (local_malloc_error_flag)
                {
                    shared_malloc_error_flag = true;
                }
            }
            // Wait for all threads to update shared flag
            #pragma omp barrier
    
            // All threads free memory if there is an error
            if (shared_malloc_error_flag)
            {
                free(interface_density_x_L);
                free(interface_density_x_R);
                free(interface_velocity_x_x_L);
                free(interface_velocity_x_x_R);
                free(interface_velocity_x_y_L);
                free(interface_velocity_x_y_R);
                free(interface_pressure_x_L);
                free(interface_pressure_x_R);
            }
    
            // Only one thread raises the error
            #pragma omp single
            {
                if (shared_malloc_error_flag)
                {
                    error_status = WRAP_RAISE_ERROR(MEMORY_ERROR, "Failed to allocate memory for interface arrays.");
                }
            }
        
            if (!shared_malloc_error_flag)
            {
                #pragma omp for
#endif
                for (int j = num_ghost_cells_side; j < (num_ghost_cells_side + num_cells_y); j++)
                {
                    ErrorStatus local_error_status;

                    /* Reconstruct interface */
                    for (int i = 0; i < (num_cells_x + 2); i++)
                    {
                        const int idx = j * total_num_cells_x + (i + num_ghost_cells_side - 1);

                        const double density_idx = density[idx];
                        const double velocity_x_idx = velocity_x[idx];
                        const double velocity_y_idx = velocity_y[idx];
                        const double pressure_idx = pressure[idx];

                        const double surface_area_x_idx = surface_area_x[i + num_ghost_cells_side - 1];
                        const double volume_idx = volume[idx];
                        const double dt_A_over_vol = dt * surface_area_x_idx / volume_idx;

                        const double sound_speed = get_sound_speed(gamma, density_idx, pressure_idx);

                        // Calculate the field difference
                        const double delta_density_L = density_idx - density[idx - 1];
                        const double delta_velocity_x_L = velocity_x_idx - velocity_x[idx - 1];
                        const double delta_velocity_y_L = velocity_y_idx - velocity_y[idx - 1];
                        const double delta_pressure_L = pressure_idx - pressure[idx - 1];

                        const double delta_density_R = density[idx + 1] - density_idx;
                        const double delta_velocity_x_R = velocity_x[idx + 1] - velocity_x_idx;
                        const double delta_velocity_y_R = velocity_y[idx + 1] - velocity_y_idx;
                        const double delta_pressure_R = pressure[idx + 1] - pressure_idx;

                        double delta_density;
                        double delta_velocity_x;
                        double delta_velocity_y;
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
                            &delta_velocity_y,
                            integrator_param,
                            delta_velocity_y_L,
                            delta_velocity_y_R
                        );
                        limit_slope(
                            &delta_pressure,
                            integrator_param,
                            delta_pressure_L,
                            delta_pressure_R
                        );

                        /**
                         *  Coefficient matrix                      
                         *          / u     rho       0       0    \
                         *   A(W) = | 0      u        0    1 / rho |
                         *          | 0      0        u       0    |
                         *          \ 0  rho a^2      0       u    /
                         *   
                         *   \overline{W}^L_i = W_i - 1/2 [I + dt / dx A(W_i)] \Delta_i                       
                         *   \overline{W}^R_i = W_i + 1/2 [I - dt / dx A(W_i)] \Delta_i
                         */
                        const double update_density_L = -0.5 * (
                            (1.0 + dt_A_over_vol * velocity_x_idx) * delta_density
                            + dt_A_over_vol * density_idx * delta_velocity_x
                            + 0.0
                            + 0.0
                        );
                        const double update_velocity_x_L = -0.5 * (
                            0.0
                            + (1.0 + dt_A_over_vol * velocity_x_idx) * delta_velocity_x
                            + 0.0
                            + (dt_A_over_vol / density_idx) * delta_pressure
                        );
                        const double update_velocity_y_L = -0.5 * (
                            0.0
                            + 0.0
                            + (1.0 + dt_A_over_vol * velocity_x_idx) * delta_velocity_y
                            + 0.0
                        );
                        const double update_pressure_L = -0.5 * (
                            0.0
                            + dt_A_over_vol * density_idx * sound_speed * sound_speed * delta_velocity_x
                            + 0.0
                            + (1.0 + dt_A_over_vol * velocity_x_idx) * delta_pressure
                        );

                        const double update_density_R = 0.5 * (
                            (1.0 - dt_A_over_vol * velocity_x_idx) * delta_density
                            - dt_A_over_vol * density_idx * delta_velocity_x
                            + 0.0
                            + 0.0
                        );
                        const double update_velocity_x_R = 0.5 * (
                            0.0
                            + (1.0 - dt_A_over_vol * velocity_x_idx) * delta_velocity_x
                            + 0.0
                            - (dt_A_over_vol / density_idx) * delta_pressure
                        );
                        const double update_velocity_y_R = 0.5 * (
                            0.0
                            + 0.0
                            + (1.0 - dt_A_over_vol * velocity_x_idx) * delta_velocity_y
                            + 0.0
                        );
                        const double update_pressure_R = 0.5 * (
                            0.0
                            - dt_A_over_vol * density_idx * sound_speed * sound_speed * delta_velocity_x
                            + 0.0
                            + (1.0 - dt_A_over_vol * velocity_x_idx) * delta_pressure
                        );

                        if (i > 0)
                        {
                            interface_density_x_R[i - 1] = density_idx + update_density_L;
                            interface_velocity_x_x_R[i - 1] = velocity_x_idx + update_velocity_x_L;
                            interface_velocity_x_y_R[i - 1] = velocity_y_idx + update_velocity_y_L;
                            interface_pressure_x_R[i - 1] = pressure_idx + update_pressure_L;
                        }
                        if (i < (num_cells_x + 1))
                        {
                            interface_density_x_L[i] = density_idx + update_density_R;
                            interface_velocity_x_x_L[i] = velocity_x_idx + update_velocity_x_R;
                            interface_velocity_x_y_L[i] = velocity_y_idx + update_velocity_y_R;
                            interface_pressure_x_L[i] = pressure_idx + update_pressure_R;
                        }
                    }

                    /* Compute fluxes */
                    for (int i = 0; i < num_interfaces_x; i++)
                    {
                        double flux_mass;
                        double flux_momentum_x;
                        double flux_momentum_y;
                        double flux_energy;

                        local_error_status = WRAP_TRACEBACK(solve_flux_2d(
                            integrator_param,
                            settings,
                            &flux_mass,
                            &flux_momentum_x,
                            &flux_momentum_y,
                            &flux_energy,
                            gamma,
                            interface_density_x_L[i],
                            interface_velocity_x_x_L[i],
                            interface_velocity_x_y_L[i],
                            interface_pressure_x_L[i],
                            interface_density_x_R[i],
                            interface_velocity_x_x_R[i],
                            interface_velocity_x_y_R[i],
                            interface_pressure_x_R[i]
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
                        const double d_rho_v = flux_momentum_y * dt;
                        const double d_energy_density = flux_energy * dt;

                        const int idx_i = num_ghost_cells_side + i;
                        const int idx_L = j * total_num_cells_x + (idx_i - 1);
                        const int idx_R = j * total_num_cells_x + idx_i;
                        const double surface_area_x_L = surface_area_x[idx_i - 1];
                        const double surface_area_x_R = surface_area_x[idx_i];

                        mass[idx_L] -= d_rho * surface_area_x_L;
                        momentum_x[idx_L] -= d_rho_u * surface_area_x_L;
                        momentum_y[idx_L] -= d_rho_v * surface_area_x_L;
                        energy[idx_L] -= d_energy_density * surface_area_x_L;

                        mass[idx_R] += d_rho * surface_area_x_R;
                        momentum_x[idx_R] += d_rho_u * surface_area_x_R;
                        momentum_y[idx_R] += d_rho_v * surface_area_x_R;
                        energy[idx_R] += d_energy_density * surface_area_x_R;
                    }
                }
#ifdef USE_OPENMP
            }
    
            free(interface_density_x_L);
            free(interface_density_x_R);
            free(interface_velocity_x_x_L);
            free(interface_velocity_x_x_R);
            free(interface_velocity_x_y_L);
            free(interface_velocity_x_y_R);
            free(interface_pressure_x_L);
            free(interface_pressure_x_R);
        }
#endif

        if (error_status.return_code != SUCCESS)
        {
            goto err_solve_flux_x;
        }

        /* Compute fluxes and update step (y-direction) */
#ifdef USE_OPENMP
        shared_malloc_error_flag = false;
        #pragma omp parallel
        {
            bool local_malloc_error_flag = false;
    
            /* Allocate memory for interface arrays */
            double *__restrict interface_density_y_B = malloc(num_interfaces_y * sizeof(double));
            double *__restrict interface_density_y_T = malloc(num_interfaces_y * sizeof(double));
            double *__restrict interface_velocity_y_x_B = malloc(num_interfaces_y * sizeof(double));
            double *__restrict interface_velocity_y_x_T = malloc(num_interfaces_y * sizeof(double));
            double *__restrict interface_velocity_y_y_B = malloc(num_interfaces_y * sizeof(double));
            double *__restrict interface_velocity_y_y_T = malloc(num_interfaces_y * sizeof(double));
            double *__restrict interface_pressure_y_B = malloc(num_interfaces_y * sizeof(double));
            double *__restrict interface_pressure_y_T = malloc(num_interfaces_y * sizeof(double));
    
            if (
                !interface_density_y_B
                || !interface_density_y_T
                || !interface_velocity_y_x_B
                || !interface_velocity_y_x_T
                || !interface_velocity_y_y_B
                || !interface_velocity_y_y_T
                || !interface_pressure_y_B
                || !interface_pressure_y_T
            )
            {
                local_malloc_error_flag = true;
            }
    
            // Update shared flag
            #pragma omp critical
            {
                if (local_malloc_error_flag)
                {
                    shared_malloc_error_flag = true;
                }
            }
    
            // Wait for all threads to update shared flag
            #pragma omp barrier
    
            // All threads free memory if there is an error
            if (shared_malloc_error_flag)
            {
                free(interface_density_y_B);
                free(interface_density_y_T);
                free(interface_velocity_y_x_B);
                free(interface_velocity_y_x_T);
                free(interface_velocity_y_y_B);
                free(interface_velocity_y_y_T);
                free(interface_pressure_y_B);
                free(interface_pressure_y_T);
            }
    
            // Only one thread raises the error
            #pragma omp single
            {
                if (shared_malloc_error_flag)
                {
                    error_status = WRAP_RAISE_ERROR(MEMORY_ERROR, "Failed to allocate memory for interface arrays.");
                }
            }
    
            if (!shared_malloc_error_flag)
            {
                #pragma omp for
#endif
                for (int i = num_ghost_cells_side; i < (num_ghost_cells_side + num_cells_x); i++)
                {
                    ErrorStatus local_error_status;

                    /* Reconstruct interface */
                    for (int j = 0; j < (num_cells_y + 2); j++)
                    {
                        const int idx = (j + num_ghost_cells_side - 1) * total_num_cells_x + i;

                        const double density_idx = density[idx];
                        const double velocity_x_idx = velocity_x[idx];
                        const double velocity_y_idx = velocity_y[idx];
                        const double pressure_idx = pressure[idx];

                        const double surface_area_y_idx = surface_area_y[j + num_ghost_cells_side - 1];
                        const double volume_idx = volume[idx];
                        const double dt_A_over_vol = dt * surface_area_y_idx / volume_idx;

                        const double sound_speed = get_sound_speed(gamma, density_idx, pressure_idx);

                        // Calculate the field difference
                        const double delta_density_B = density_idx - density[idx - total_num_cells_x];
                        const double delta_velocity_x_B = velocity_x_idx - velocity_x[idx - total_num_cells_x];
                        const double delta_velocity_y_B = velocity_y_idx - velocity_y[idx - total_num_cells_x];
                        const double delta_pressure_B = pressure_idx - pressure[idx - total_num_cells_x];

                        const double delta_density_T = density[idx + total_num_cells_x] - density_idx;
                        const double delta_velocity_x_T = velocity_x[idx + total_num_cells_x] - velocity_x_idx;
                        const double delta_velocity_y_T = velocity_y[idx + total_num_cells_x] - velocity_y_idx;
                        const double delta_pressure_T = pressure[idx + total_num_cells_x] - pressure_idx;

                        double delta_density;
                        double delta_velocity_x;
                        double delta_velocity_y;
                        double delta_pressure;

                        /* Slope limiting */
                        limit_slope(
                            &delta_density,
                            integrator_param,
                            delta_density_B,
                            delta_density_T
                        );
                        limit_slope(
                            &delta_velocity_x,
                            integrator_param,
                            delta_velocity_x_B,
                            delta_velocity_x_T
                        );
                        limit_slope(
                            &delta_velocity_y,
                            integrator_param,
                            delta_velocity_y_B,
                            delta_velocity_y_T
                        );
                        limit_slope(
                            &delta_pressure,
                            integrator_param,
                            delta_pressure_B,
                            delta_pressure_T
                        );

                        /**
                         *  Coefficient matrix                      
                         *          / v     rho       0       0    \
                         *   B(W) = | 0      v        0       0    |
                         *          | 0      0        v    1 / rho |
                         *          \ 0      0     rho a^2    v    /
                         *   
                         *   \overline{W}^B_i = W_i - 1/2 [I + dt / dx B(W_i)] \Delta_i                       
                         *   \overline{W}^T_i = W_i + 1/2 [I - dt / dx B(W_i)] \Delta_i
                         */
                        const double update_density_B = -0.5 * (
                            (1.0 + dt_A_over_vol * velocity_y_idx) * delta_density
                            + dt_A_over_vol * density_idx * delta_velocity_x
                            + 0.0
                            + 0.0
                        );
                        const double update_velocity_x_B = -0.5 * (
                            0.0
                            + (1.0 + dt_A_over_vol * velocity_y_idx) * delta_velocity_x
                            + 0.0
                            + 0.0
                        );
                        const double update_velocity_y_B = -0.5 * (
                            0.0
                            + 0.0
                            + (1.0 + dt_A_over_vol * velocity_y_idx) * delta_velocity_y
                            + (dt_A_over_vol / density_idx) * delta_pressure
                        );
                        const double update_pressure_B = -0.5 * (
                            0.0
                            + 0.0
                            + dt_A_over_vol * density_idx * sound_speed * sound_speed * delta_velocity_y
                            + (1.0 + dt_A_over_vol * velocity_y_idx) * delta_pressure
                        );

                        const double update_density_T = 0.5 * (
                            (1.0 - dt_A_over_vol * velocity_y_idx) * delta_density
                            - dt_A_over_vol * density_idx * delta_velocity_x
                            + 0.0
                            + 0.0
                        );
                        const double update_velocity_x_T = 0.5 * (
                            0.0
                            + (1.0 - dt_A_over_vol * velocity_y_idx) * delta_velocity_x
                            + 0.0
                            + 0.0
                        );
                        const double update_velocity_y_T = 0.5 * (
                            0.0
                            + 0.0
                            + (1.0 - dt_A_over_vol * velocity_y_idx) * delta_velocity_y
                            - (dt_A_over_vol / density_idx) * delta_pressure
                        );
                        const double update_pressure_T = 0.5 * (
                            0.0
                            + 0.0
                            - dt_A_over_vol * density_idx * sound_speed * sound_speed * delta_velocity_y
                            + (1.0 - dt_A_over_vol * velocity_y_idx) * delta_pressure
                        );

                        if (j > 0)
                        {
                            interface_density_y_T[j - 1] = density_idx + update_density_B;
                            interface_velocity_y_x_T[j - 1] = velocity_x_idx + update_velocity_x_B;
                            interface_velocity_y_y_T[j - 1] = velocity_y_idx + update_velocity_y_B;
                            interface_pressure_y_T[j - 1] = pressure_idx + update_pressure_B;
                        }
                        if (j < (num_cells_y + 1))
                        {
                            interface_density_y_B[j] = density_idx + update_density_T;
                            interface_velocity_y_x_B[j] = velocity_x_idx + update_velocity_x_T;
                            interface_velocity_y_y_B[j] = velocity_y_idx + update_velocity_y_T;
                            interface_pressure_y_B[j] = pressure_idx + update_pressure_T;
                        }
                    }

                    /* Compute fluxes */
                    for (int j = 0; j < num_interfaces_y; j++)
                    {
                        double flux_mass;
                        double flux_momentum_y;
                        double flux_momentum_x;
                        double flux_energy;

                        local_error_status = WRAP_TRACEBACK(solve_flux_2d(
                            integrator_param,
                            settings,
                            &flux_mass,
                            &flux_momentum_y,
                            &flux_momentum_x,
                            &flux_energy,
                            gamma,
                            interface_density_y_B[j],
                            interface_velocity_y_y_B[j],
                            interface_velocity_y_x_B[j],
                            interface_pressure_y_B[j],
                            interface_density_y_T[j],
                            interface_velocity_y_y_T[j],
                            interface_velocity_y_x_T[j],
                            interface_pressure_y_T[j]
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
                        const double d_rho_v = flux_momentum_y * dt;
                        const double d_energy_density = flux_energy * dt;

                        const int idx_j = num_ghost_cells_side + j;
                        const int idx_B = (idx_j - 1) * total_num_cells_x + i;
                        const int idx_T = idx_j * total_num_cells_x + i;
                        const double surface_area_y_B = surface_area_y[idx_j - 1];
                        const double surface_area_y_T = surface_area_y[idx_j];

                        mass[idx_B] -= d_rho * surface_area_y_B;
                        momentum_x[idx_B] -= d_rho_u * surface_area_y_B;
                        momentum_y[idx_B] -= d_rho_v * surface_area_y_B;
                        energy[idx_B] -= d_energy_density * surface_area_y_B;

                        mass[idx_T] += d_rho * surface_area_y_T;
                        momentum_x[idx_T] += d_rho_u * surface_area_y_T;
                        momentum_y[idx_T] += d_rho_v * surface_area_y_T;
                        energy[idx_T] += d_energy_density * surface_area_y_T;
                    }
                }
#ifdef USE_OPENMP
            }
    
            free(interface_density_y_B);
            free(interface_density_y_T);
            free(interface_velocity_y_x_B);
            free(interface_velocity_y_x_T);
            free(interface_velocity_y_y_B);
            free(interface_velocity_y_y_T);
            free(interface_pressure_y_B);
            free(interface_pressure_y_T);
        }
#endif
        if (error_status.return_code != SUCCESS)
        {
            goto err_solve_flux_y;
        }

        error_status = WRAP_TRACEBACK(convert_conserved_to_primitive(system));
        if (error_status.return_code != SUCCESS)
        {
            goto err_convert_conserved_to_primitive;
        }
        error_status = WRAP_TRACEBACK(set_boundary_condition(boundary_condition_param, system));
        if (error_status.return_code != SUCCESS)
        {
            goto err_set_boundary_condition;
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

#ifndef USE_OPENMP
    free(interface_density_x_L);
    free(interface_density_x_R);
    free(interface_velocity_x_x_L);
    free(interface_velocity_x_x_R);
    free(interface_velocity_x_y_L);
    free(interface_velocity_x_y_R);
    free(interface_pressure_x_L);
    free(interface_pressure_x_R);

    free(interface_density_y_B);
    free(interface_density_y_T);
    free(interface_velocity_y_x_B);
    free(interface_velocity_y_x_T);
    free(interface_velocity_y_y_B);
    free(interface_velocity_y_y_T);
    free(interface_pressure_y_B);
    free(interface_pressure_y_T);
#endif

    return make_success_error_status();

err_store_snapshot:
err_set_boundary_condition:
err_convert_conserved_to_primitive:
err_solve_flux_y:
err_solve_flux_x:
err_dt_zero:
    if (!no_progress_bar)
    {
        update_progress_bar(&progress_bar_param, *t_ptr, true);
    }
err_store_first_snapshot:

#ifndef USE_OPENMP
err_memory_interface:
    free(interface_density_x_L);
    free(interface_density_x_R);
    free(interface_velocity_x_x_L);
    free(interface_velocity_x_x_R);
    free(interface_velocity_x_y_L);
    free(interface_velocity_x_y_R);
    free(interface_pressure_x_L);
    free(interface_pressure_x_R);

    free(interface_density_y_B);
    free(interface_density_y_T);
    free(interface_velocity_y_x_B);
    free(interface_velocity_y_x_T);
    free(interface_velocity_y_y_B);
    free(interface_velocity_y_y_T);
    free(interface_pressure_y_B);
    free(interface_pressure_y_T);
#endif

err_coord_sys:
    return error_status;
}

