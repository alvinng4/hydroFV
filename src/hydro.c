/**
 * \file hydro.c
 * 
 * \brief Main functions for launching the hydrodynamics simulation.
 * 
 * \author Ching-Yin Ng
 * \date 2025-03-08
 */

#include <stdlib.h>

#include "hydro.h"
#include "integrator.h"
#include "riemann_solver.h"

WIN32DLL_API ErrorStatus launch_simulation(
    System *system,
    IntegratorParam *integrator_param,
    StoringParam *storing_param,
    Settings *settings,
    SimulationParam *simulation_param
)
{
    ErrorStatus error_status;

    error_status = WRAP_TRACEBACK(get_integrator_flag(integrator_param));
    if (error_status.return_code != SUCCESS)
    {
        goto error;
    }

    error_status = WRAP_TRACEBACK(get_riemann_solver_flag(integrator_param));
    if (error_status.return_code != SUCCESS)
    {
        goto error;
    }

    // return_code = get_storing_method_flag(storing_param);
    // if (return_code != SUCCESS)
    // {
    //     goto error;
    // }

    error_status = WRAP_TRACEBACK(integrator_launch_simulation(
        system,
        integrator_param,
        storing_param,
        settings,
        simulation_param
    ));
    if (error_status.return_code != SUCCESS)
    {
        goto error;
    }

    return error_status;

error:
    return error_status;
}
