/**
 * \file hydro.c
 * 
 * \brief Main functions for launching the hydrodynamics simulation.
 * 
 * \author Ching-Yin Ng
 * \date 2025-03-25
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "hydro.h"
#include "hydro_time.h"
#include "integrator.h"
#include "storing.h"


IN_FILE ErrorStatus final_check(
    BoundaryConditionParam *__restrict boundary_condition_param,
    System *__restrict system,
    IntegratorParam *__restrict integrator_param,
    StoringParam *__restrict storing_param,
    Settings *__restrict settings,
    SimulationParam *__restrict simulation_param,
    SimulationStatus *__restrict simulation_status
);

IN_FILE void print_simulation_parameters(
    const BoundaryConditionParam *__restrict boundary_condition_param,
    const System *__restrict system,
    const IntegratorParam *__restrict integrator_param,
    const StoringParam *__restrict storing_param,
    const Settings *__restrict settings,
    const SimulationParam *__restrict simulation_param
);

ErrorStatus launch_simulation(
    BoundaryConditionParam *__restrict boundary_condition_param,
    System *__restrict system,
    IntegratorParam *__restrict integrator_param,
    StoringParam *__restrict storing_param,
    Settings *__restrict settings,
    SimulationParam *__restrict simulation_param,
    SimulationStatus *__restrict simulation_status
)
{
    ErrorStatus error_status;

    error_status = WRAP_TRACEBACK(finalize_boundary_condition_param(system, boundary_condition_param));
    if (error_status.return_code != SUCCESS)
    {
        goto error;
    }

    error_status = WRAP_TRACEBACK(finalize_integrator_param(integrator_param));
    if (error_status.return_code != SUCCESS)
    {
        goto error;
    }

    error_status = WRAP_TRACEBACK(finalize_storing_param(storing_param));
    if (error_status.return_code != SUCCESS)
    {
        goto error;
    }

    error_status = WRAP_TRACEBACK(final_check(
        boundary_condition_param,
        system,
        integrator_param,
        storing_param,
        settings,
        simulation_param,
        simulation_status
    ));
    if (error_status.return_code != SUCCESS)
    {
        goto error;
    }

    if (settings->verbose >= 1)
    {
        print_simulation_parameters(
            boundary_condition_param,
            system,
            integrator_param,
            storing_param,
            settings,
            simulation_param
        );
    }

    double start;
    double end;

    if (settings->verbose >= 1)
    {
        printf("Launching simulation...\n");
        start = hydro_get_current_time();
    }
    error_status = WRAP_TRACEBACK(integrator_launch_simulation(
        boundary_condition_param,
        system,
        integrator_param,
        storing_param,
        settings,
        simulation_param,
        simulation_status
    ));
    if (error_status.return_code != SUCCESS)
    {
        goto error;
    }
    if (settings->verbose >= 1)
    {
        end = hydro_get_current_time();
        printf("Simulation completed in %.3g seconds. Number of steps: %lld\n", end - start, simulation_status->num_steps);
    }

    return error_status;

error:
    return error_status;
}

IN_FILE ErrorStatus final_check(
    BoundaryConditionParam *__restrict boundary_condition_param,
    System *__restrict system,
    IntegratorParam *__restrict integrator_param,
    StoringParam *__restrict storing_param,
    Settings *__restrict settings,
    SimulationParam *__restrict simulation_param,
    SimulationStatus *__restrict simulation_status
)
{
    ErrorStatus error_status;

    (void) boundary_condition_param;
    (void) storing_param;
    (void) settings;
    (void) simulation_param;
    (void) simulation_status;

    /* Check number of ghost cells */
    switch (integrator_param->reconstruction_flag_)
    {
        case RECONSTRUCTION_PIECEWISE_CONSTANT:
            if (system->num_ghost_cells_side < 1)
            {
                error_status = WRAP_RAISE_ERROR(VALUE_ERROR, "Number of ghost cells must be at least 1.");
                goto error;
            }
            break;
        case RECONSTRUCTION_PIECEWISE_LINEAR:
            if (system->num_ghost_cells_side < 2)
            {
                error_status = WRAP_RAISE_ERROR(VALUE_ERROR, "Number of ghost cells must be at least 2.");
                goto error;
            }
            break;
        case RECONSTRUCTION_PIECEWISE_PARABOLIC:
            if (system->num_ghost_cells_side < 3)
            {
                error_status = WRAP_RAISE_ERROR(VALUE_ERROR, "Number of ghost cells must be at least 3.");
                goto error;
            }
            break;
        default:
            error_status = WRAP_RAISE_ERROR(VALUE_ERROR, "Reconstruction flag not recognized.");
            goto error;
    }

    /* Check coordinate system */
    switch (integrator_param->integrator_flag_)
    {
        case INTEGRATOR_RANDOM_CHOICE_1D: case INTEGRATOR_GODUNOV_FIRST_ORDER_1D:
            if (
                system->coord_sys_flag_ != COORD_SYS_CARTESIAN_1D
                && system->coord_sys_flag_ != COORD_SYS_CYLINDRICAL_1D
                && system->coord_sys_flag_ != COORD_SYS_SPHERICAL_1D
            )
            {
                size_t error_message_size = (
                    strlen("Wrong coordinate system. Supported coordinate system for integrator \"\": \"cartesian_1d\", \"cylindrical_1d\" and \"spherical_1d\", got: \"\"")
                    + strlen(integrator_param->integrator)
                    + strlen(system->coord_sys)
                    + 1 // For null terminator
                );
                char *__restrict error_message = malloc(error_message_size * sizeof(char));
                if (!error_message)
                {
                    error_status = WRAP_RAISE_ERROR(MEMORY_ERROR, "Failed to allocate memory for error message.");
                    goto error;
                }
                snprintf(
                    error_message,
                    error_message_size,
                    "Wrong coordinate system. Supported coordinate system for integrator \"%s\": \"cartesian_1d\", \"cylindrical_1d\" and \"spherical_1d\", got: \"%s\"",
                    integrator_param->integrator,
                    system->coord_sys
                );
                error_status = WRAP_RAISE_ERROR(VALUE_ERROR, error_message);
                free(error_message);
                goto error;
            }
            break;
        case INTEGRATOR_GODUNOV_FIRST_ORDER_2D:
            if (system->coord_sys_flag_ != COORD_SYS_CARTESIAN_2D)
            {
                size_t error_message_size = (
                    strlen("Wrong coordinate system. Supported coordinate system for integrator \"\": \"cartesian_2d\", got: \"\"")
                    + strlen(integrator_param->integrator)
                    + strlen(system->coord_sys)
                    + 1 // For null terminator
                );
                char *__restrict error_message = malloc(error_message_size * sizeof(char));
                if (!error_message)
                {
                    error_status = WRAP_RAISE_ERROR(MEMORY_ERROR, "Failed to allocate memory for error message.");
                    goto error;
                }
                snprintf(
                    error_message,
                    error_message_size,
                    "Wrong coordinate system. Supported coordinate system for integrator \"%s\": \"cartesian_2d\", got: \"%s\"",
                    integrator_param->integrator,
                    system->coord_sys
                );
                error_status = WRAP_RAISE_ERROR(VALUE_ERROR, error_message);
                free(error_message);
                goto error;
            }
            break;
        case INTEGRATOR_GODUNOV_FIRST_ORDER_3D:
            if (system->coord_sys_flag_ != COORD_SYS_CARTESIAN_3D)
            {
                size_t error_message_size = (
                    strlen("Wrong coordinate system. Supported coordinate system for integrator \"\": \"cartesian_3d\", got: \"\"")
                    + strlen(integrator_param->integrator)
                    + strlen(system->coord_sys)
                    + 1 // For null terminator
                );
                char *__restrict error_message = malloc(error_message_size * sizeof(char));
                if (!error_message)
                {
                    error_status = WRAP_RAISE_ERROR(MEMORY_ERROR, "Failed to allocate memory for error message.");
                    goto error;
                }
                snprintf(
                    error_message,
                    error_message_size,
                    "Wrong coordinate system. Supported coordinate system for integrator \"%s\": \"cartesian_3d\", got: \"%s\"",
                    integrator_param->integrator,
                    system->coord_sys
                );
                error_status = WRAP_RAISE_ERROR(VALUE_ERROR, error_message);
                free(error_message);
                goto error;
            }
            break;
        default:
            error_status = WRAP_RAISE_ERROR(VALUE_ERROR, "Integrator flag not recognized.");
            goto error;
    }

    /* Check Riemann solver */
    if (integrator_param->integrator_flag_ == INTEGRATOR_RANDOM_CHOICE_1D)
    {
        if (integrator_param->riemann_solver_flag_ != RIEMANN_SOLVER_EXACT)
        {
            size_t error_message_size = (
                strlen("Supported Riemann solver for random choice method 1D: \"riemann_solver_exact\", got: \"\"") 
                + strlen(integrator_param->riemann_solver) 
                + 1 // For null terminator
            );
            char *__restrict error_message = malloc(error_message_size * sizeof(char));
            if (!error_message)
            {
                error_status = WRAP_RAISE_ERROR(MEMORY_ERROR, "Failed to allocate memory for error message.");
                goto error;
            }
            snprintf(
                error_message,
                error_message_size,
                "Supported Riemann solver for random choice method 1D: \"riemann_solver_exact\", got: \"%s\"",
                integrator_param->riemann_solver
            );
            error_status = WRAP_RAISE_ERROR(VALUE_ERROR, error_message);
            free(error_message);
            goto error;
        }
    }

    return make_success_error_status();

error:
    return error_status;
}

IN_FILE void print_simulation_parameters(
    const BoundaryConditionParam *__restrict boundary_condition_param,
    const System *__restrict system,
    const IntegratorParam *__restrict integrator_param,
    const StoringParam *__restrict storing_param,
    const Settings *__restrict settings,
    const SimulationParam *__restrict simulation_param
)
{
    fputs("=============== Simulation Parameters ===============\n", stdout);
    fputs("System:\n", stdout);
    printf("    Coordinate system: %s\n", system->coord_sys);
    printf("    Gamma: %.3g\n", system->gamma);

    switch (system->coord_sys_flag_)
    {
        case COORD_SYS_CARTESIAN_1D: case COORD_SYS_CYLINDRICAL_1D: case COORD_SYS_SPHERICAL_1D:    
            printf("    Domain x: [%.3g, %.3g]\n", system->x_min, system->x_max);
            printf("    Number of cells: [%d]\n", system->num_cells_x);
            printf("    Number of ghost cells per side: %d\n", system->num_ghost_cells_side);
            break;
        case COORD_SYS_CARTESIAN_2D:
            printf("    gravity: %.3g\n", system->gravity);
            printf("    Domain x: [%.3g, %.3g]\n", system->x_min, system->x_max);
            printf("    Domain y: [%.3g, %.3g]\n", system->y_min, system->y_max);
            printf("    Number of cells: [%d, %d]\n", system->num_cells_x, system->num_cells_y);
            printf("    Number of ghost cells per side: %d\n", system->num_ghost_cells_side);
            break;
        case COORD_SYS_CARTESIAN_3D:
            printf("    gravity: %.3g\n", system->gravity);
            printf("    Domain x: [%.3g, %.3g]\n", system->x_min, system->x_max);
            printf("    Domain y: [%.3g, %.3g]\n", system->y_min, system->y_max);
            printf("    Domain z: [%.3g, %.3g]\n", system->z_min, system->z_max);
            printf("    Number of cells: [%d, %d, %d]\n", system->num_cells_x, system->num_cells_y, system->num_cells_z);
            printf("    Number of ghost cells per side: %d\n", system->num_ghost_cells_side);
            break;
    }

    printf("Integrator:\n");
    printf("    Integrator: %s\n", integrator_param->integrator);
    printf("    Riemann solver: %s\n", integrator_param->riemann_solver);
    
    switch (integrator_param->integrator_flag_)
    {
        case INTEGRATOR_RANDOM_CHOICE_1D:
            break;
        case INTEGRATOR_GODUNOV_FIRST_ORDER_1D:
        case INTEGRATOR_GODUNOV_FIRST_ORDER_2D:
        case INTEGRATOR_GODUNOV_FIRST_ORDER_3D:
            printf("    Reconstruction: %s\n", integrator_param->reconstruction);
            printf("    Slope limiter: %s\n", integrator_param->reconstruction_limiter);
            printf("    Time integrator: %s\n", integrator_param->time_integrator);
            break;
    }

    printf("    CFL: %.3g\n", integrator_param->cfl);
    printf("    CFL initial shrink factor: %.3g\n", integrator_param->cfl_initial_shrink_factor);
    printf("    Initial CFL shrink steps: %d\n", integrator_param->cfl_initial_shrink_num_steps);
    printf("    Tolerance for Riemann solver: %.3g\n", integrator_param->tol);

    printf("Boundary conditions:\n");
    switch (system->coord_sys_flag_)
    {
        case COORD_SYS_CARTESIAN_1D: case COORD_SYS_CYLINDRICAL_1D: case COORD_SYS_SPHERICAL_1D:
            printf("    Boundary condition x_min: %s\n", boundary_condition_param->boundary_condition_x_min);
            printf("    Boundary condition x_max: %s\n", boundary_condition_param->boundary_condition_x_max);
            break;
        case COORD_SYS_CARTESIAN_2D:
            printf("    Boundary condition x_min: %s\n", boundary_condition_param->boundary_condition_x_min);
            printf("    Boundary condition x_max: %s\n", boundary_condition_param->boundary_condition_x_max);
            printf("    Boundary condition y_min: %s\n", boundary_condition_param->boundary_condition_y_min);
            printf("    Boundary condition y_max: %s\n", boundary_condition_param->boundary_condition_y_max);
            break;
        case COORD_SYS_CARTESIAN_3D:
            printf("    Boundary condition x_min: %s\n", boundary_condition_param->boundary_condition_x_min);
            printf("    Boundary condition x_max: %s\n", boundary_condition_param->boundary_condition_x_max);
            printf("    Boundary condition y_min: %s\n", boundary_condition_param->boundary_condition_y_min);
            printf("    Boundary condition y_max: %s\n", boundary_condition_param->boundary_condition_y_max);
            printf("    Boundary condition z_min: %s\n", boundary_condition_param->boundary_condition_z_min);
            printf("    Boundary condition z_max: %s\n", boundary_condition_param->boundary_condition_z_max);
            break;
    }

    printf("Storing:\n");
    if (storing_param->is_storing)
    {
        printf("    Output directory: %s\n", storing_param->output_dir);
        printf("    Storing initial system: %s\n", storing_param->store_initial ? "true" : "false");
        printf("    Storing interval: %.3g\n", storing_param->storing_interval);
    }
    else
    {
        printf("    Storing disabled\n");
    }
    
    printf("Settings:\n");
    printf("    Verbose: %d\n", settings->verbose);
    printf("    Progress bar: %s\n", settings->no_progress_bar ? "disabled" : "enabled");

    printf("Simulation:\n");
    printf("    tf: %.3g\n", simulation_param->tf);

    fputs("=====================================================\n", stdout);
    fflush(stdout);
}
