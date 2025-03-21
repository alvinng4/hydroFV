/**
 * \file storing.h
 * 
 * \brief Header file for storing snapshots related functions.
 * 
 * \author Ching-Yin Ng
 * \date 2025-03-21
 */

#ifndef STORING_H
#define STORING_H

#include "common.h"
#include "system.h"

/**
 * \brief Get uninitialized storing parameters struct.
 * 
 * \return StoringParam Uninitialized storing parameters struct.
 */
StoringParam get_new_storing_param(void);

/**
 * \brief Finalize and check the storing parameters.
 * 
 * \param storing_param Pointer to the storing parameters.
 * 
 * \return ErrorStatus.
 */
ErrorStatus finalize_storing_param(StoringParam *__restrict storing_param);

/**
 * \brief Store a snapshot of the hydrodynamic system.
 * 
 * \param boundary_condition_param Pointer to the boundary condition parameters.
 * \param system Pointer to the hydrodynamic system.
 * \param integrator_param Pointer to the integrator parameters.
 * \param simulation_status Pointer to the simulation status.
 * \param storing_param Pointer to the storing parameters.
 * 
 * \return ErrorStatus.
 */
ErrorStatus store_snapshot(
    const BoundaryConditionParam *__restrict boundary_condition_param,
    const System *__restrict system,
    const IntegratorParam *__restrict integrator_param,
    const SimulationStatus *__restrict simulation_status,
    StoringParam *__restrict storing_param
);

#endif
