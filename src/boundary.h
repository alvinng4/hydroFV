/**
 * \file boundary.h
 * 
 * \brief Header file for the boundary conditions.
 * 
 * \author Ching-Yin Ng
 * \date 2025-03-21
 */

#ifndef BOUNDARY_H
#define BOUNDARY_H

#include "common.h"
#include "system.h"

// Boundary condition functions pointer prototypes
typedef void (*set_boundary_condition_1d_func)(
    double *__restrict density,
    double *__restrict velocity_x,
    double *__restrict pressure,
    const int num_ghost_cells_side,
    const int num_cells_x
);

typedef void (*set_boundary_condition_cartesian_2d_func)(
    double *__restrict density,
    double *__restrict velocity_x,
    double *__restrict velocity_y,
    double *__restrict pressure,
    const int num_ghost_cells_side,
    const int num_cells_x,
    const int num_cells_y
);

typedef struct BoundaryConditionParam
{
    const char *boundary_condition_x_min;
    const char *boundary_condition_x_max;
    const char *boundary_condition_y_min;
    const char *boundary_condition_y_max;
    const char *boundary_condition_z_min;
    const char *boundary_condition_z_max;
    set_boundary_condition_1d_func set_boundary_condition_1d_x_min;
    set_boundary_condition_1d_func set_boundary_condition_1d_x_max;
    set_boundary_condition_cartesian_2d_func set_boundary_condition_cartesian_2d_x_min;
    set_boundary_condition_cartesian_2d_func set_boundary_condition_cartesian_2d_x_max;
    set_boundary_condition_cartesian_2d_func set_boundary_condition_cartesian_2d_y_min;
    set_boundary_condition_cartesian_2d_func set_boundary_condition_cartesian_2d_y_max;

    int boundary_condition_flag_x_min_;
    int boundary_condition_flag_x_max_;
    int boundary_condition_flag_y_min_;
    int boundary_condition_flag_y_max_;
    int boundary_condition_flag_z_min_;
    int boundary_condition_flag_z_max_;
} BoundaryConditionParam;

// Boundary condition flags
#define BOUNDARY_CONDITION_NONE 1
#define BOUNDARY_CONDITION_REFLECTIVE 2
#define BOUNDARY_CONDITION_TRANSMISSIVE 3
#define BOUNDARY_CONDITION_PERIODIC 4
#define BOUNDARY_CONDITION_CUSTOM 5

/**
 * \brief Get a new boundary condition parameter struct.
 * 
 * \return Boundary condition parameter struct.
 */
BoundaryConditionParam get_new_boundary_condition_param(void);

ErrorStatus finalize_boundary_condition_param(
    const System *__restrict system,
    BoundaryConditionParam *__restrict boundary_condition_param
);

/**
 * \brief Set the boundary conditions.
 * 
 * \param boundary_condition_param Pointer to the boundary condition parameter struct.
 * \param system Pointer to the system struct.
 *
 * \return Error status.
 */
ErrorStatus set_boundary_condition(
    const BoundaryConditionParam *__restrict boundary_condition_param,
    System *__restrict system
);

ErrorStatus set_boundary_condition_1d(
    const BoundaryConditionParam *__restrict boundary_condition_param,
    double *__restrict density,
    double *__restrict velocity_x,
    double *__restrict pressure,
    const int num_ghost_cells_side,
    const int num_cells_x
);

ErrorStatus set_boundary_condition_cartesian_2d(
    const BoundaryConditionParam *__restrict boundary_condition_param,
    double *__restrict density,
    double *__restrict velocity_x,
    double *__restrict velocity_y,
    double *__restrict pressure,
    const int num_ghost_cells_side,
    const int num_cells_x,
    const int num_cells_y
);

#endif
