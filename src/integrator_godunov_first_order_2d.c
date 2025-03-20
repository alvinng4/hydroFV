/**
 * \file integrator_godunov_first_order_1d.c
 * 
 * \brief 2D First-order Godunov scheme for the Euler equations.
 * 
 * \author Ching-Yin Ng
 * \date 2025-03-20
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


IN_FILE void reconstruct_interface_2d(
    const IntegratorParam *__restrict integrator_param,
    const double *__restrict density,
    const double *__restrict velocity_x,
    const double *__restrict velocity_y,
    const double *__restrict pressure,
    double *__restrict interface_density_L,
    double *__restrict interface_density_R,
    double *__restrict interface_velocity_x_x_L,
    double *__restrict interface_velocity_x_x_R,
    double *__restrict interface_velocity_x_y_L,
    double *__restrict interface_velocity_x_y_R,
    double *__restrict interface_pressure_L,
    double *__restrict interface_pressure_R,
    const int num_cells_x,
    const int num_ghost_cells_side
)
{
    reconstruct_cell_interface(
        integrator_param,
        density,
        interface_density_L,
        interface_density_R,
        num_cells_x,
        num_ghost_cells_side
    );
    reconstruct_cell_interface(
        integrator_param,
        velocity_x,
        interface_velocity_x_x_L,
        interface_velocity_x_x_R,
        num_cells_x,
        num_ghost_cells_side
    );
    reconstruct_cell_interface(
        integrator_param,
        velocity_y,
        interface_velocity_x_y_L,
        interface_velocity_x_y_R,
        num_cells_x,
        num_ghost_cells_side
    );
    reconstruct_cell_interface(
        integrator_param,
        pressure,
        interface_pressure_L,
        interface_pressure_R,
        num_cells_x,
        num_ghost_cells_side
    );
}

IN_FILE ErrorStatus time_integrator_euler_2d(
    IntegratorParam *__restrict integrator_param,
    Settings *__restrict settings,
    double *__restrict density,
    double *__restrict velocity_x,
    double *__restrict velocity_y,
    double *__restrict pressure,
    double *__restrict mass,
    double *__restrict momentum_x,
    double *__restrict momentum_y,
    double *__restrict energy,
    double *__restrict surface_area_x,
    double *__restrict surface_area_y,
    double *__restrict volume,
    const int num_cells_x,
    const int num_cells_y,
    const int num_ghost_cells_side,
    const int num_interfaces_x,
    const int num_interfaces_y,
    const double dt,
    const double gamma,
    const int boundary_condition_flag_x_min,
    const int boundary_condition_flag_x_max,
    const int boundary_condition_flag_y_min,
    const int boundary_condition_flag_y_max
)
{
    ErrorStatus error_status = make_success_error_status();
    const int total_num_cells_x = num_cells_x + 2 * num_ghost_cells_side;
    const int total_num_cells_y = num_cells_y + 2 * num_ghost_cells_side;

#ifndef USE_OPENMP
    /* Allocate memory for temporary arrays */
    double *__restrict temp_density_y = malloc(total_num_cells_y * sizeof(double));
    double *__restrict temp_velocity_y_x = malloc(total_num_cells_y * sizeof(double));
    double *__restrict temp_velocity_y_y = malloc(total_num_cells_y * sizeof(double));
    double *__restrict temp_pressure_y = malloc(total_num_cells_y * sizeof(double));
    if (
        !temp_density_y
        || !temp_velocity_y_x
        || !temp_velocity_y_y
        || !temp_pressure_y
    )
    {
        error_status = WRAP_RAISE_ERROR(MEMORY_ERROR, "Failed to allocate memory for temporary arrays.");
        goto err_memory_temp;
    }

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
                /* Reconstruct interface */
                reconstruct_interface_2d(
                    integrator_param,
                    &(density[j * total_num_cells_x]),
                    &(velocity_x[j * total_num_cells_x]),
                    &(velocity_y[j * total_num_cells_x]),
                    &(pressure[j * total_num_cells_x]),
                    interface_density_x_L,
                    interface_density_x_R,
                    interface_velocity_x_x_L,
                    interface_velocity_x_x_R,
                    interface_velocity_x_y_L,
                    interface_velocity_x_y_R,
                    interface_pressure_x_L,
                    interface_pressure_x_R,
                    num_cells_x,
                    num_ghost_cells_side
                );

                /* Compute fluxes and update step (x-direction) */
                for (int i = 0; i < num_interfaces_x; i++)
                {
                    ErrorStatus local_error_status;

                    /* x-direction */
                    double flux_mass_x;
                    double flux_momentum_x_x;
                    double flux_momentum_x_y;
                    double flux_energy_x;

                    local_error_status = WRAP_TRACEBACK(solve_flux_2d(
                        integrator_param,
                        settings,
                        &flux_mass_x,
                        &flux_momentum_x_x,
                        &flux_momentum_x_y,
                        &flux_energy_x,
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

                    const int idx_i = num_ghost_cells_side + i;
                    mass[j * total_num_cells_x + (idx_i - 1)] -= dt * flux_mass_x * surface_area_x[idx_i - 1];
                    momentum_x[j * total_num_cells_x + (idx_i - 1)] -= dt * flux_momentum_x_x * surface_area_x[idx_i - 1];
                    momentum_y[j * total_num_cells_x + (idx_i - 1)] -= dt * flux_momentum_x_y * surface_area_x[idx_i - 1];
                    energy[j * total_num_cells_x + (idx_i - 1)] -= dt * flux_energy_x * surface_area_x[idx_i - 1];

                    mass[j * total_num_cells_x + idx_i] += dt * flux_mass_x * surface_area_x[idx_i];
                    momentum_x[j * total_num_cells_x + idx_i] += dt * flux_momentum_x_x * surface_area_x[idx_i];
                    momentum_y[j * total_num_cells_x + idx_i] += dt * flux_momentum_x_y * surface_area_x[idx_i];
                    energy[j * total_num_cells_x + idx_i] += dt * flux_energy_x * surface_area_x[idx_i];
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
            goto err_compute_fluxes_x;
        }

        /* Compute fluxes and update step (y-direction) */
#ifdef USE_OPENMP
    shared_malloc_error_flag = false;
    #pragma omp parallel
    {
        bool local_malloc_error_flag = false;

        /* Allocate memory for temporary arrays */
        double *__restrict temp_density_y = malloc(total_num_cells_y * sizeof(double));
        double *__restrict temp_velocity_y_x = malloc(total_num_cells_y * sizeof(double));
        double *__restrict temp_velocity_y_y = malloc(total_num_cells_y * sizeof(double));
        double *__restrict temp_pressure_y = malloc(total_num_cells_y * sizeof(double));

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
            !temp_density_y
            || !temp_velocity_y_x
            || !temp_velocity_y_y
            || !temp_pressure_y
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
            free(temp_density_y);
            free(temp_velocity_y_x);
            free(temp_velocity_y_y);
            free(temp_pressure_y);

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
                error_status = WRAP_RAISE_ERROR(MEMORY_ERROR, "Failed to allocate memory for temporary arrays.");
            }
        }

        if (!shared_malloc_error_flag)
        {
            #pragma omp for
#endif
            for (int i = num_ghost_cells_side; i < (num_ghost_cells_side + num_cells_x); i++)
            {
                // Reconstruct interface
                for (int j = 0; j < total_num_cells_y; j++)
                {
                    temp_density_y[j] = density[j * total_num_cells_x + i];
                    temp_velocity_y_x[j] = velocity_x[j * total_num_cells_x + i];
                    temp_velocity_y_y[j] = velocity_y[j * total_num_cells_x + i];
                    temp_pressure_y[j] = pressure[j * total_num_cells_x + i];
                }
                reconstruct_interface_2d(
                    integrator_param,
                    temp_density_y,
                    temp_velocity_y_y,
                    temp_velocity_y_x,
                    temp_pressure_y,
                    interface_density_y_B,
                    interface_density_y_T,
                    interface_velocity_y_y_B,
                    interface_velocity_y_y_T,
                    interface_velocity_y_x_B,
                    interface_velocity_y_x_T,
                    interface_pressure_y_B,
                    interface_pressure_y_T,
                    num_cells_y,
                    num_ghost_cells_side
                );

                // Compute fluxes and update step
                for (int j = 0; j < num_interfaces_y; j++)
                {
                    ErrorStatus local_error_status;

                    /* y-direction */
                    double flux_mass_y;
                    double flux_momentum_y_x;
                    double flux_momentum_y_y;
                    double flux_energy_y;

                    local_error_status = WRAP_TRACEBACK(solve_flux_2d(
                        integrator_param,
                        settings,
                        &flux_mass_y,
                        &flux_momentum_y_y,
                        &flux_momentum_y_x,
                        &flux_energy_y,
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
                        error_status = local_error_status;
                    }

                    const int idx_j = num_ghost_cells_side + j;

                    mass[(idx_j - 1) * total_num_cells_x + i] -= dt * flux_mass_y * surface_area_y[idx_j - 1];
                    momentum_x[(idx_j - 1) * total_num_cells_x + i] -= dt * flux_momentum_y_x * surface_area_y[idx_j - 1];
                    momentum_y[(idx_j - 1) * total_num_cells_x + i] -= dt * flux_momentum_y_y * surface_area_y[idx_j - 1];
                    energy[(idx_j - 1) * total_num_cells_x + i] -= dt * flux_energy_y * surface_area_y[idx_j - 1];

                    mass[idx_j * total_num_cells_x + i] += dt * flux_mass_y * surface_area_y[idx_j];
                    momentum_x[idx_j * total_num_cells_x + i] += dt * flux_momentum_y_x * surface_area_y[idx_j];
                    momentum_y[idx_j * total_num_cells_x + i] += dt * flux_momentum_y_y * surface_area_y[idx_j];
                    energy[idx_j * total_num_cells_x + i] += dt * flux_energy_y * surface_area_y[idx_j];
                }
            }
#ifdef USE_OPENMP
        }

        free(temp_density_y);
        free(temp_velocity_y_x);
        free(temp_velocity_y_y);
        free(temp_pressure_y);

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
        goto err_compute_fluxes_y;
    }

    /* Convert conserved to primitive variables */
    convert_conserved_to_primitive_2d(
        num_cells_x,
        num_cells_y,
        num_ghost_cells_side,
        gamma,
        volume,
        mass,
        momentum_x,
        momentum_y,
        energy,
        density,
        velocity_x,
        velocity_y,
        pressure
    );

    /* Set boundary conditions */
    error_status = set_boundary_condition_cartesian_2d(
        density,
        velocity_x,
        velocity_y,
        pressure,
        num_ghost_cells_side,
        num_cells_x,
        num_cells_y,
        boundary_condition_flag_x_min,
        boundary_condition_flag_x_max,
        boundary_condition_flag_y_min,
        boundary_condition_flag_y_max
    );

#ifndef USE_OPENMP
    free(temp_density_y);
    free(temp_velocity_y_x);
    free(temp_velocity_y_y);
    free(temp_pressure_y);

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

    return error_status;

err_compute_fluxes_y:
err_compute_fluxes_x:

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
err_memory_temp:
    free(temp_density_y);
    free(temp_velocity_y_x);
    free(temp_velocity_y_y);
    free(temp_pressure_y);
#endif

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
    ErrorStatus error_status = make_success_error_status();

    /* System parameters */
    const double gamma = system->gamma;
    const int num_ghost_cells_side = system->num_ghost_cells_side;
    const int num_cells_x = system->num_cells_x;
    const int num_cells_y = system->num_cells_y;
    const int num_interfaces_x = num_cells_x + 1;
    const int num_interfaces_y = num_cells_y + 1;
    const int boundary_condition_flag_x_min = system->boundary_condition_flag_x_min_;
    const int boundary_condition_flag_x_max = system->boundary_condition_flag_x_max_;
    const int boundary_condition_flag_y_min = system->boundary_condition_flag_y_min_;
    const int boundary_condition_flag_y_max = system->boundary_condition_flag_y_max_;

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
    double *__restrict velocity_y = system->velocity_y_;
    double *__restrict pressure = system->pressure_;
    double *__restrict mass = system->mass_;
    double *__restrict momentum_x = system->momentum_x_;
    double *__restrict momentum_y = system->momentum_y_;
    double *__restrict energy = system->energy_;
    double *__restrict surface_area_x = system->surface_area_x_;
    double *__restrict surface_area_y = system->surface_area_y_;
    double *__restrict volume = system->volume_;

    /* Time integrator function */
    ErrorStatus (*time_integrator_2d_func)(
        IntegratorParam *__restrict integrator_param,
        Settings *__restrict settings,
        double *__restrict density,
        double *__restrict velocity_x,
        double *__restrict velocity_y,
        double *__restrict pressure,
        double *__restrict mass,
        double *__restrict momentum_x,
        double *__restrict momentum_y,
        double *__restrict energy,
        double *__restrict surface_area_x,
        double *__restrict surface_area_y,
        double *__restrict volume,
        const int num_cells_x,
        const int num_cells_y,
        const int num_ghost_cells_side,
        const int num_interfaces_x,
        const int num_interfaces_y,
        const double dt,
        const double gamma,
        const int boundary_condition_flag_x_min,
        const int boundary_condition_flag_x_max,
        const int boundary_condition_flag_y_min,
        const int boundary_condition_flag_y_max
    );

    switch (integrator_param->time_integrator_flag_)
    {
        case TIME_INTEGRATOR_EULER:
            time_integrator_2d_func = time_integrator_euler_2d;
            break;
        // case TIME_INTEGRATOR_SSP_RK2:
        //     time_integrator_2d_func = time_integrator_ssp_rk2_2d;
        //     break;
        // case TIME_INTEGRATOR_SSP_RK3:
        //     time_integrator_2d_func = time_integrator_ssp_rk3_2d;
        //     break;
        default:
            error_status = WRAP_RAISE_ERROR(VALUE_ERROR, "Time integrator not recognized.");
            return error_status;
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

        /* Advance step */
        error_status = time_integrator_2d_func(
            integrator_param,
            settings,
            density,
            velocity_x,
            velocity_y,
            pressure,
            mass,
            momentum_x,
            momentum_y,
            energy,
            surface_area_x,
            surface_area_y,
            volume,
            num_cells_x,
            num_cells_y,
            num_ghost_cells_side,
            num_interfaces_x,
            num_interfaces_y,
            dt,
            gamma,
            boundary_condition_flag_x_min,
            boundary_condition_flag_x_max,
            boundary_condition_flag_y_min,
            boundary_condition_flag_y_max
        );
        if (error_status.return_code != SUCCESS)
        {
            goto err_advance_step;
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
err_advance_step:
err_dt_zero:
    if (!no_progress_bar)
    {
        update_progress_bar(&progress_bar_param, *t_ptr, true);
    }
err_store_first_snapshot:

    return error_status;
}
