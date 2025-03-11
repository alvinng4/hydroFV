/**
 * \file hydro.h
 * 
 * \brief Header file for the hydrodynamic module.
 * 
 * \author Ching-Yin Ng
 * \date 2025-03-11
 */

#ifndef HYDRO_H
#define HYDRO_H

/* Common definitions */
#include "common.h"

/* Error handling */
#include "error.h"

/* Integrators */
#include "integrator.h"

/* Riemann solver */
#include "riemann_solver.h"

/* Definitions for storing simulation data */
#include "storing.h"

/* Definitions related to the hydrodynamic system */
#include "system.h"


/**
 * \brief Launch the hydrodynamics simulation.
 * 
 * \param system Pointer to the hydrodynamic system.
 * \param integrator_param Pointer to the integrator parameters.
 * \param storing_param Pointer to the storing parameters.
 * \param settings Pointer to the settings.
 * \param simulation_param Pointer to the simulation parameters.
 * \param simulation_status Pointer to the simulation status.
 * 
 * \return ErrorStatus.
 */
ErrorStatus launch_simulation(
    System *__restrict system,
    IntegratorParam *__restrict integrator_param,
    StoringParam *__restrict storing_param,
    Settings *__restrict settings,
    SimulationParam *__restrict simulation_param,
    SimulationStatus *__restrict simulation_status
);

#endif
