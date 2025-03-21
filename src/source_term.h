/**
 * \file source_term.h
 * 
 * \brief Header file for the source term calculation
 * 
 * \author Ching-Yin Ng
 * \date 2025-03-21
 */

#ifndef SOURCE_TERM_H
#define SOURCE_TERM_H

#include "hydro.h"

/**
 * \brief Add the geometry source term.
 * 
 * \param system Pointer to the system
 * \param dt Time step
 */
ErrorStatus add_geometry_source_term(
    const BoundaryConditionParam *__restrict boundary_condition_param,
    System *__restrict system,
    const double dt
);

ErrorStatus add_gravity_source_term_2d(
    const BoundaryConditionParam *__restrict boundary_condition_param,
    System *__restrict system,
    const double dt
);

#endif
