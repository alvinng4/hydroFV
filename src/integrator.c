/**
 * \file integrator.c
 * 
 * \brief Integrator related functions for the hydrodynamics simulation.
 * 
 * \author Ching-Yin Ng
 * \date 2025-03-08
 */

#include <string.h>

#include "hydro.h"
#include "integrator.h"

ErrorStatus get_integrator_flag(IntegratorParam *__restrict integrator_param)
{
    if (strcmp(integrator_param->integrator, "godunov_first_order_1d") == 0)
    {
        integrator_param->integrator_flag_ = INTEGRATOR_GODUNOV_FIRST_ORDER_1D;
        return make_success_error_status();
    }
    else if (strcmp(integrator_param->integrator, "random_choice_1d") == 0)
    {
        integrator_param->integrator_flag_ = INTEGRATOR_RANDOM_CHOICE_1D;
        return make_success_error_status();
    }
    else
    {
        return WRAP_RAISE_ERROR(VALUE_ERROR, "Integrator not recognized.");
    }
}

ErrorStatus integrator_launch_simulation(
    System *__restrict system,
    IntegratorParam *__restrict integrator_param,
    StoringParam *__restrict storing_param,
    Settings *__restrict settings,
    SimulationParam *__restrict simulation_param
)
{
    switch (integrator_param->integrator_flag_)
    {
        case INTEGRATOR_GODUNOV_FIRST_ORDER_1D:
            return WRAP_TRACEBACK(godunov_first_order_1d(system, integrator_param, storing_param, settings, simulation_param));
        case INTEGRATOR_RANDOM_CHOICE_1D:
            return WRAP_TRACEBACK(random_choice_1d(system, integrator_param, storing_param, settings, simulation_param));
        default:
            return WRAP_RAISE_ERROR(VALUE_ERROR, "Integrator flag not recognized.");
    }
}