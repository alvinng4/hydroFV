/**
 * \file system.h
 * 
 * \brief Header file for definition and prototypes related to the hydrodynamics system.
 * 
 * \author Ching-Yin Ng
 * \date 2025-03-08
 */

#ifndef SYSTEM_H
#define SYSTEM_H

#include "common.h"

// Coordinate system flags
#define COORD_SYS_CARTESIAN_1D 1
#define COORD_SYS_CARTESIAN_2D 2
#define COORD_SYS_CARTESIAN_3D 3
#define COORD_SYS_CYLINDRICAL_1D 4
#define COORD_SYS_SPHERICAL_1D 5

// Boundary condition flags
#define BOUNDARY_CONDITION_NONE 1
#define BOUNDARY_CONDITION_REFLECTIVE 2
#define BOUNDARY_CONDITION_TRANSMISSIVE 3
#define BOUNDARY_CONDITION_PERIODIC 4

typedef struct System
{
    const char *coord_sys;
    const char *boundary_condition_x_min;
    const char *boundary_condition_x_max;
    const char *boundary_condition_y_min;
    const char *boundary_condition_y_max;
    const char *boundary_condition_z_min;
    const char *boundary_condition_z_max;
    real gamma;
    real x_min;
    real x_max;
    real y_min;
    real y_max;
    real z_min;
    real z_max;
    int num_cells_x;
    int num_cells_y;
    int num_cells_z;
    int num_ghost_cells_side;

    int coord_sys_flag_;
    int boundary_condition_flag_x_min_;
    int boundary_condition_flag_x_max_;
    int boundary_condition_flag_y_min_;
    int boundary_condition_flag_y_max_;
    int boundary_condition_flag_z_min_;
    int boundary_condition_flag_z_max_;
    real dx_;
    real dy_;
    real dz_;
    real *density_;
    real *velocity_;
    real *pressure_;
    real *mass_;
    real *momentum_;
    real *energy_;
    real *mid_points_x_;
    real *mid_points_y_;
    real *mid_points_z_;
    real *volume_;
} System;


/**
 * \brief Get a new system struct.
 * 
 * \return The system struct.
 */
System get_new_system_struct();

/**
 * \brief Initialize the hidden variables for the system struct.
 * 
 * \param system The system struct.
 * 
 * \return SUCCESS if successful
 * \return return_code if an error occurred
 */
ErrorStatus system_init(System *__restrict system);

/**
 * \brief Free the memory allocated for the system struct.
 * 
 * \param system The system struct.
 */
void free_system_memory(System *__restrict system);

/**
 * \brief Set the boundary conditions.
 * 
 * \param system The system struct.
 * 
 * \return SUCCESS if successful
 * \return return_code if an error occurred
 */
ErrorStatus set_boundary_condition(System *__restrict system);

/**
 * \brief Convert the conserved variables to primitive variables.
 * 
 * \param system The system struct.
 * 
 * \return SUCCESS if successful
 * \return return_code if an error occurred
 */
ErrorStatus convert_conserved_to_primitive(System *__restrict system);


/**
 * \brief Convert the primitive variables to conserved variables.
 * 
 * \param system The system struct.
 * 
 * \return SUCCESS if successful
 * \return return_code if an error occurred
 */
ErrorStatus convert_primitive_to_conserved(System *__restrict system);

#endif
