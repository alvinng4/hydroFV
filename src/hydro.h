/**
 * \file hydro.h
 * 
 * \brief Header file for the hydrodynamic module.
 * 
 * \author Ching-Yin Ng
 * \date 2025-03-06
 */

#ifndef HYDRO_H
#define HYDRO_H

/* Common definitions */
#include "common.h"

/* Error handling */
#include "error.h"

/* Definitions related to the hydrodynamic system */
#include "system.h"

ErrorStatus launch_simulation(
    System *system,
    IntegratorParam *integrator_param,
    StoringParam *storing_param,
    Settings *settings,
    SimulationParam *simulation_param
);

#endif