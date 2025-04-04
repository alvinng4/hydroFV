/**
 * \file integrator.c
 * 
 * \brief Integrator related functions for the hydrodynamics simulation.
 * 
 * \author Ching-Yin Ng
 * \date 2025-03-21
 */

#include <stdio.h>
#include <string.h>

#include "hydro.h"


IntegratorParam get_new_integrator_param(void)
{
    IntegratorParam integrator_param;
    integrator_param.integrator = NULL;
    integrator_param.riemann_solver = NULL;
    integrator_param.slope_limiter = NULL;
    integrator_param.cfl = 0.5;
    integrator_param.cfl_initial_shrink_factor = 0.2;
    integrator_param.cfl_initial_shrink_num_steps = 10;
    integrator_param.tol = 1e-6;
    integrator_param.integrator_flag_ = -1;
    integrator_param.riemann_solver_flag_ = -1;
    integrator_param.slope_limiter_flag_ = -1;
    return integrator_param;
}

IN_FILE ErrorStatus get_integrator_flag(IntegratorParam *__restrict integrator_param)
{
    /* Integrator */
    if (!integrator_param->integrator)
    {
        return WRAP_RAISE_ERROR(POINTER_ERROR, "Integrator is not set.");
    }

    if (strcmp(integrator_param->integrator, "random_choice_1d") == 0)
    {
        integrator_param->integrator_flag_ = INTEGRATOR_RANDOM_CHOICE_1D;
    }
    else if (strcmp(integrator_param->integrator, "godunov_first_order_1d") == 0)
    {
        integrator_param->integrator_flag_ = INTEGRATOR_GODUNOV_FIRST_ORDER_1D;
    }
    else if (strcmp(integrator_param->integrator, "godunov_first_order_2d") == 0)
    {
        integrator_param->integrator_flag_ = INTEGRATOR_GODUNOV_FIRST_ORDER_2D;
    }
    else if (strcmp(integrator_param->integrator, "muscl_hancock_1d") == 0)
    {
        integrator_param->integrator_flag_ = INTEGRATOR_MUSCL_HANCOCK_1D;
    }
    else if (strcmp(integrator_param->integrator, "muscl_hancock_2d") == 0)
    {
        integrator_param->integrator_flag_ = INTEGRATOR_MUSCL_HANCOCK_2D;
    }
    else
    {
        return WRAP_RAISE_ERROR(VALUE_ERROR, "Integrator not recognized.");
    }

    return make_success_error_status();
}

ErrorStatus finalize_integrator_param(IntegratorParam *__restrict integrator_param)
{
    ErrorStatus error_status;

    error_status = get_integrator_flag(integrator_param);
    if (error_status.return_code != SUCCESS)
    {
        return error_status;
    }
    error_status = get_riemann_solver_flag(integrator_param);
    if (error_status.return_code != SUCCESS)
    {
        return error_status;
    }
    error_status = get_slope_limiter_flag(integrator_param);
    if (error_status.return_code != SUCCESS)
    {
        return error_status;
    }

    if (integrator_param->cfl <= 0.0)
    {
        return WRAP_RAISE_ERROR(VALUE_ERROR, "CFL number must be positive.");
    }
    if (integrator_param->cfl_initial_shrink_factor <= 0.0)
    {
        return WRAP_RAISE_ERROR(VALUE_ERROR, "CFL initial shrink factor must be positive.");
    }
    if (integrator_param->cfl_initial_shrink_num_steps <= 0)
    {
        return WRAP_RAISE_ERROR(VALUE_ERROR, "CFL initial shrink number of steps must be positive.");
    }
    if (integrator_param->tol <= 0.0)
    {
        return WRAP_RAISE_ERROR(VALUE_ERROR, "Tolerance must be positive.");
    }

    return make_success_error_status();
}

ErrorStatus integrator_launch_simulation(
    BoundaryConditionParam *__restrict boundary_condition_param,
    System *__restrict system,
    IntegratorParam *__restrict integrator_param,
    StoringParam *__restrict storing_param,
    Settings *__restrict settings,
    SimulationParam *__restrict simulation_param,
    SimulationStatus *__restrict simulation_status
)
{
    switch (integrator_param->integrator_flag_)
    {
        case INTEGRATOR_RANDOM_CHOICE_1D:
            return WRAP_TRACEBACK(random_choice_1d(
                boundary_condition_param, system, integrator_param, storing_param, settings, simulation_param, simulation_status
            ));
        case INTEGRATOR_GODUNOV_FIRST_ORDER_1D:
            return WRAP_TRACEBACK(godunov_first_order_1d(
                boundary_condition_param, system, integrator_param, storing_param, settings, simulation_param, simulation_status
            ));
        case INTEGRATOR_GODUNOV_FIRST_ORDER_2D:
            return WRAP_TRACEBACK(godunov_first_order_2d(
                boundary_condition_param, system, integrator_param, storing_param, settings, simulation_param, simulation_status
            ));
        case INTEGRATOR_MUSCL_HANCOCK_1D:
            return WRAP_TRACEBACK(muscl_hancock_1d(
                boundary_condition_param, system, integrator_param, storing_param, settings, simulation_param, simulation_status
            ));
        case INTEGRATOR_MUSCL_HANCOCK_2D:
            return WRAP_TRACEBACK(muscl_hancock_2d(
                boundary_condition_param, system, integrator_param, storing_param, settings, simulation_param, simulation_status
            ));
        default:
            return WRAP_RAISE_ERROR(VALUE_ERROR, "Integrator flag not recognized.");
    }
}
