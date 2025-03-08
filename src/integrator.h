/**
 * \file integrator.h
 * 
 * \brief Header file for integrator related definitions and prototypes.
 * 
 * \author Ching-Yin Ng
 * \date 2025-03-08
 */

#ifndef INTEGRATOR_H
#define INTEGRATOR_H

#include "hydro.h"

#define INTEGRATOR_GODUNOV_FIRST_ORDER_1D 1
#define INTEGRATOR_RANDOM_CHOICE_1D 2

/**
 * \brief Get the integrator flag from the integrator name.
 * 
 * \param integrator_param The integrator parameter.
 * 
 * \retval SUCCESS if successful.
 * \retval ERROR_UNKNOWN_INTEGRATOR if the integrator name is unknown.
 */
ErrorStatus get_integrator_flag(IntegratorParam *__restrict integrator_param);

/**
 * \brief Launch the simulation with the specified integrator.
 * 
 * \param system Hydrodynamical system.
 * \param integrator_param Integrator parameters.
 * \param storing_param Storing parameters.
 * \param settings Settings.
 * \param simulation_param Simulation parameters.
 */
ErrorStatus integrator_launch_simulation(
    System *__restrict system,
    IntegratorParam *__restrict integrator_param,
    StoringParam *__restrict storing_param,
    Settings *__restrict settings,
    SimulationParam *__restrict simulation_param
);

ErrorStatus godunov_first_order_1d(
    System *__restrict system,
    IntegratorParam *__restrict integrator_param,
    StoringParam *__restrict storing_param,
    Settings *__restrict settings,
    SimulationParam *__restrict simulation_param
);

ErrorStatus random_choice_1d(
    System *__restrict system,
    IntegratorParam *__restrict integrator_param,
    StoringParam *__restrict storing_param,
    Settings *__restrict settings,
    SimulationParam *__restrict simulation_param
);

#endif
