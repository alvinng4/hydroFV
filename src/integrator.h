/**
 * \file integrator.h
 * 
 * \brief Header file for integrator related definitions and prototypes.
 * 
 * \author Ching-Yin Ng
 * \date 2025-03-11
 */

#ifndef INTEGRATOR_H
#define INTEGRATOR_H

#include "common.h"
#include "system.h"

#define INTEGRATOR_GODUNOV_FIRST_ORDER_1D 1
#define INTEGRATOR_GODUNOV_FIRST_ORDER_2D 2
#define INTEGRATOR_GODUNOV_FIRST_ORDER_3D 3
#define INTEGRATOR_RANDOM_CHOICE_1D 4

/**
 * \brief Get uninitialized integrator parameters struct.
 * 
 * \return IntegratorParam Uninitialized integrator parameters struct.
 */
IntegratorParam get_new_integrator_param(void);

/**
 * \brief Finalize and check the integrator parameters.
 * 
 * \param integrator_param Pointer to the integrator parameters.
 * 
 * \return Error status.
 */
ErrorStatus finalize_integrator_param(IntegratorParam *__restrict integrator_param);

/**
 * \brief Launch the simulation with the specified integrator.
 * 
 * \param system Pointer to the hydrodynamic system.
 * \param integrator_param Pointer to the integrator parameters.
 * \param storing_param Pointer to the storing parameters.
 * \param settings Pointer to the settings.
 * \param simulation_param Pointer to the simulation parameters.
 * \param simulation_status Pointer to the simulation status.
 * 
 * \return Error status.
 */
ErrorStatus integrator_launch_simulation(
    System *__restrict system,
    IntegratorParam *__restrict integrator_param,
    StoringParam *__restrict storing_param,
    Settings *__restrict settings,
    SimulationParam *__restrict simulation_param,
    SimulationStatus *__restrict simulation_status
);

/**
 * \brief First-order Godunov scheme for the 1D Euler equations.
 * 
 * \param system Pointer to the hydrodynamic system.
 * \param integrator_param Pointer to the integrator parameters.
 * \param storing_param Pointer to the storing parameters.
 * \param settings Pointer to the settings.
 * \param simulation_param Pointer to the simulation parameters.
 * \param simulation_status Pointer to the simulation status.
 * 
 * \return Error status.
 */
ErrorStatus godunov_first_order_1d(
    System *__restrict system,
    IntegratorParam *__restrict integrator_param,
    StoringParam *__restrict storing_param,
    Settings *__restrict settings,
    SimulationParam *__restrict simulation_param,
    SimulationStatus *__restrict simulation_status
);

/**
 * \brief First-order Godunov scheme for the 2D Euler equations.
 * 
 * \param system Pointer to the hydrodynamic system.
 * \param integrator_param Pointer to the integrator parameters.
 * \param storing_param Pointer to the storing parameters.
 * \param settings Pointer to the settings.
 * \param simulation_param Pointer to the simulation parameters.
 * \param simulation_status Pointer to the simulation status.
 * 
 * \return Error status.
 */
ErrorStatus godunov_first_order_2d(
    System *__restrict system,
    IntegratorParam *__restrict integrator_param,
    StoringParam *__restrict storing_param,
    Settings *__restrict settings,
    SimulationParam *__restrict simulation_param,
    SimulationStatus *__restrict simulation_status
);

/**
 * \brief Random choice method for the 1D Euler equations.
 * 
 * \param system Pointer to the hydrodynamic system.
 * \param integrator_param Pointer to the integrator parameters.
 * \param storing_param Pointer to the storing parameters.
 * \param settings Pointer to the settings.
 * \param simulation_param Pointer to the simulation parameters.
 * \param simulation_status Pointer to the simulation status.
 * 
 * \return Error status.
 */
ErrorStatus random_choice_1d(
    System *__restrict system,
    IntegratorParam *__restrict integrator_param,
    StoringParam *__restrict storing_param,
    Settings *__restrict settings,
    SimulationParam *__restrict simulation_param,
    SimulationStatus *__restrict simulation_status
);

#endif
