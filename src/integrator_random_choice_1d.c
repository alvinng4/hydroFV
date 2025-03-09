/**
 * \file integrator_godunov_first_order_1d.c
 * 
 * \brief First-order Godunov scheme for the 1D Euler equations.
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

#define VAN_DER_CORPUT_SEQUENCE_K1 5
#define VAN_DER_CORPUT_SEQUENCE_K2 3

IN_FILE real get_van_der_corput_sequence(int64 n)
{
    real result = 0;
    int base = VAN_DER_CORPUT_SEQUENCE_K1;
    real bk = 1.0 / (real) base;
    while (n > 0)
    {
        const int a_i = n % base;
        const int A_i = (VAN_DER_CORPUT_SEQUENCE_K2 * a_i) % VAN_DER_CORPUT_SEQUENCE_K1;
        result += A_i * bk;

        n /= base;
        bk /= (real) base;
    }
    return result;
}

WIN32DLL_API ErrorStatus random_choice_1d(
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

    const real gamma = system->gamma;
    real *__restrict density = system->density_;
    real *__restrict velocity_x = system->velocity_x_;
    real *__restrict pressure = system->pressure_;
    const real dx = system->dx_;
    const int num_cells = system->num_cells_x;
    const int num_ghost_cells_side = system->num_ghost_cells_side;
    const int num_total_cells = num_cells + 2 * num_ghost_cells_side;

    const real cfl = integrator_param->cfl;
    const real cfl_initial_shrink_factor = integrator_param->cfl_initial_shrink_factor;
    const int num_steps_shrink = integrator_param->num_steps_shrink;

    const bool no_progress_bar = settings->no_progress_bar;

    const real tf = simulation_param->tf;

    const int num_interfaces = num_cells + 1;

    real t = 0.0;

    /* Allocate memory */
    real *restrict temp_density = malloc(num_total_cells * sizeof(real));
    real *restrict temp_velocity_x = malloc(num_total_cells * sizeof(real));
    real *restrict temp_pressure = malloc(num_total_cells * sizeof(real));
    if (!temp_density || !temp_velocity_x || !temp_pressure)
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

        /* Update step */
        real theta = get_van_der_corput_sequence(count + 1);
        memcpy(temp_density, density, num_total_cells * sizeof(real));
        memcpy(temp_velocity_x, velocity_x, num_total_cells * sizeof(real));
        memcpy(temp_pressure, pressure, num_total_cells * sizeof(real));
        for (int i = num_ghost_cells_side; i < (num_ghost_cells_side + num_interfaces); i++)
        {
            real rho_L;
            real u_L;
            real p_L;
            real rho_R;
            real u_R;
            real p_R;
            real speed;
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

            error_status = WRAP_TRACEBACK(solve_exact_1d(
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
            if (error_status.return_code != SUCCESS)
            {
                goto err_solve_flux;
            }
        }

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

        t += dt;
        count++;

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

    free(temp_density);
    free(temp_velocity_x);
    free(temp_pressure);

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
err_memory:
    free(temp_density);
    free(temp_velocity_x);
    free(temp_pressure);
err_riemann_solver:
err_coord_sys:
    return error_status;
}
