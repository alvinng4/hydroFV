/**
 * \file integrator_random_choice_1d.c
 * 
 * \brief Random choice method for the 1D Euler equations.
 * 
 * \author Ching-Yin Ng
 * \date 2025-03-11
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
#include "riemann_solver.h"
#include "source_term.h"
#include "utils.h"

#define VAN_DER_CORPUT_SEQUENCE_K1 5
#define VAN_DER_CORPUT_SEQUENCE_K2 3

IN_FILE double get_van_der_corput_sequence(int64 n)
{
    double result = 0;
    int base = VAN_DER_CORPUT_SEQUENCE_K1;
    double bk = 1.0 / (double) base;
    while (n > 0)
    {
        const int a_i = n % base;
        const int A_i = (VAN_DER_CORPUT_SEQUENCE_K2 * a_i) % VAN_DER_CORPUT_SEQUENCE_K1;
        result += A_i * bk;

        n /= base;
        bk /= (double) base;
    }
    return result;
}

ErrorStatus random_choice_1d(
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

    if (integrator_param->riemann_solver_flag_ != RIEMANN_SOLVER_EXACT)
    {
        size_t error_message_size = strlen(
            "Wrong Riemann solver for random choice method 1D. Supported Riemann solver: \"riemann_solver_exact\", got: \"\""
        ) + strlen(integrator_param->riemann_solver) + 1;
        char *__restrict error_message = malloc(error_message_size * sizeof(char));
        if (!error_message)
        {
            error_status = WRAP_RAISE_ERROR(MEMORY_ERROR, "Memory allocation for error message failed.");
            goto err_riemann_solver;
        }
        snprintf(
            error_message,
            error_message_size,
            "Wrong Riemann solver for random choice method 1D. Supported Riemann solver: \"riemann_solver_exact\", got: \"%s\"",
            integrator_param->riemann_solver
        );
        error_status = WRAP_RAISE_ERROR(VALUE_ERROR, error_message);
        goto err_riemann_solver;
    }

    if (integrator_param->cfl > 0.5)
    {
        size_t error_message_size = strlen("For random choice method, CFL value should be less than or equal to 0.5. Got: ") + 1 + 128;
        char error_message[error_message_size];
        snprintf(error_message, error_message_size, "For random choice method, CFL value should be less than or equal to 0.5. Got: %.3g", integrator_param->cfl);
        WRAP_RAISE_WARNING(error_message);
    }

    bool is_compute_geometry_source_term = true;
    if (system->coord_sys_flag_ == COORD_SYS_CARTESIAN_1D)
    {
        is_compute_geometry_source_term = false;
    }

    /* Arrays */
    double *__restrict density = system->density_;
    double *__restrict velocity_x = system->velocity_x_;
    double *__restrict pressure = system->pressure_;

    /* System parameters */
    const double gamma = system->gamma;
    const int num_cells = system->num_cells_x;
    const int num_ghost_cells_side = system->num_ghost_cells_side;
    const int num_total_cells = num_cells + 2 * num_ghost_cells_side;
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

    /* Allocate memory */
    double *__restrict temp_density = malloc(num_total_cells * sizeof(double));
    double *__restrict temp_velocity_x = malloc(num_total_cells * sizeof(double));
    double *__restrict temp_pressure = malloc(num_total_cells * sizeof(double));
    if (!temp_density || !temp_velocity_x || !temp_pressure)
    {
        error_status = WRAP_RAISE_ERROR(MEMORY_ERROR, "Memory allocation failed.");
        goto err_memory;
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

        /* Update step */
        const double theta = get_van_der_corput_sequence(*num_steps_ptr + 1);
        memcpy(temp_density, density, num_total_cells * sizeof(double));
        memcpy(temp_velocity_x, velocity_x, num_total_cells * sizeof(double));
        memcpy(temp_pressure, pressure, num_total_cells * sizeof(double));
        error_status = make_success_error_status();
#ifdef USE_OPENMP
        #pragma omp parallel for
#endif
        for (int i = num_ghost_cells_side; i < (num_ghost_cells_side + num_cells + 1); i++)
        {
            double rho_L;
            double u_L;
            double p_L;
            double rho_R;
            double u_R;
            double p_R;
            double speed;
            if (theta <= 0.5)
            {
                rho_L = temp_density[i - 1];
                u_L = temp_velocity_x[i - 1];
                p_L = temp_pressure[i - 1];
                rho_R = temp_density[i];
                u_R = temp_velocity_x[i];
                p_R = temp_pressure[i];
                speed = theta * dx / dt;
            }
            else
            {
                rho_L = temp_density[i];
                u_L = temp_velocity_x[i];
                p_L = temp_pressure[i];
                rho_R = temp_density[i + 1];
                u_R = temp_velocity_x[i + 1];
                p_R = temp_pressure[i + 1];
                speed = (theta - 1.0) * dx / dt;
            }

            ErrorStatus local_error_status = WRAP_TRACEBACK(solve_exact_1d(
                &density[i],
                &velocity_x[i],
                &pressure[i],
                gamma,
                rho_L,
                u_L,
                p_L,
                rho_R,
                u_R,
                p_R,
                integrator_param->tol,
                speed,
                settings->verbose
            ));
            if (local_error_status.return_code != SUCCESS)
            {
                error_status = local_error_status;
#ifndef USE_OPENMP
                goto err_solve_flux;
#endif
            }
        }

#ifdef USE_OPENMP
        if (error_status.return_code != SUCCESS)
        {
            goto err_solve_flux;
        }
#endif

        error_status = WRAP_TRACEBACK(set_boundary_condition(system));
        if (error_status.return_code != SUCCESS)
        {
            goto err_set_boundary_condition;
        }

        error_status = WRAP_TRACEBACK(convert_primitive_to_conserved(system));
        if (error_status.return_code != SUCCESS)
        {
            goto err_convert_conserved_to_primitive;
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

    free(temp_density);
    free(temp_velocity_x);
    free(temp_pressure);

    return make_success_error_status();

err_store_snapshot:
err_set_boundary_condition:
err_convert_conserved_to_primitive:
err_compute_geometry_source_term:
err_solve_flux:
err_dt_zero:
    if (!no_progress_bar)
    {
        update_progress_bar(&progress_bar_param, *t_ptr, true);
    }
err_store_first_snapshot:
err_memory:
    free(temp_density);
    free(temp_velocity_x);
    free(temp_pressure);
err_riemann_solver:
err_coord_sys:
    return error_status;
}
