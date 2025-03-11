/**
 * \file hydro.c
 * 
 * \brief Main functions for launching the hydrodynamics simulation.
 * 
 * \author Ching-Yin Ng
 * \date 2025-03-11
 */

#include "hydro.h"
#include "integrator.h"
#include "storing.h"

ErrorStatus launch_simulation(
    System *__restrict system,
    IntegratorParam *__restrict integrator_param,
    StoringParam *__restrict storing_param,
    Settings *__restrict settings,
    SimulationParam *__restrict simulation_param,
    SimulationStatus *__restrict simulation_status
)
{
    ErrorStatus error_status;

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

    error_status = WRAP_TRACEBACK(integrator_launch_simulation(
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

    return error_status;

error:
    return error_status;
}
