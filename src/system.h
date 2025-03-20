/**
 * \file system.h
 * 
 * \brief Header file for definition and prototypes related to the hydrodynamics system.
 * 
 * \author Ching-Yin Ng
 * \date 2025-03-20
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
    double gamma;
    double x_min;
    double x_max;
    double y_min;
    double y_max;
    double z_min;
    double z_max;
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
    double dx_;
    double dy_;
    double dz_;
    double *density_;
    double *velocity_x_;
    double *velocity_y_;
    double *velocity_z_;
    double *pressure_;
    double *mass_;
    double *momentum_x_;
    double *momentum_y_;
    double *momentum_z_;
    double *energy_;
    double *mid_points_x_;
    double *mid_points_y_;
    double *mid_points_z_;
    double *surface_area_x_;
    double *surface_area_y_;
    double *surface_area_z_;
    double *volume_;
} System;


/**
 * \brief Get new system struct.
 * 
 * \return New system struct.
 */
System get_new_system_struct(void);

/**
 * \brief Initialize the hidden variables for the system struct.
 * 
 * \param system Pointer to the system struct.
 * 
 * \return Error status.
 */
ErrorStatus system_init(System *__restrict system);

/**
 * \brief Free the memory allocated for the system struct.
 * 
 * \param system Pointer to the system struct.
 */
void free_system_memory(System *__restrict system);

ErrorStatus set_boundary_condition_1d(
    double *__restrict density,
    double *__restrict velocity_x,
    double *__restrict pressure,
    const int num_ghost_cells_side,
    const int num_cells_x,
    const int boundary_condition_flag_x_min,
    const int boundary_condition_flag_x_max
);

ErrorStatus set_boundary_condition_cartesian_2d(
    double *__restrict density,
    double *__restrict velocity_x,
    double *__restrict velocity_y,
    double *__restrict pressure,
    const int num_ghost_cells_side,
    const int num_cells_x,
    const int num_cells_y,
    const int boundary_condition_flag_x_min,
    const int boundary_condition_flag_x_max,
    const int boundary_condition_flag_y_min,
    const int boundary_condition_flag_y_max
);

/**
 * \brief Set the boundary conditions.
 * 
 * \param system Pointer to the system struct.
 *
 * \return Error status.
 */
ErrorStatus set_boundary_condition(System *__restrict system);

void convert_conserved_to_primitive_1d(
    const int num_cells_x,
    const int num_ghost_cells_side,
    const double gamma,
    const double *__restrict volume,
    const double *__restrict mass,
    const double *__restrict momentum_x,
    const double *__restrict energy,
    double *__restrict density,
    double *__restrict velocity_x,
    double *__restrict pressure
);

void convert_conserved_to_primitive_2d(
    const int num_cells_x,
    const int num_cells_y,
    const int num_ghost_cells_side,
    const double gamma,
    const double *__restrict volume,
    const double *__restrict mass,
    const double *__restrict momentum_x,
    const double *__restrict momentum_y,
    const double *__restrict energy,
    double *__restrict density,
    double *__restrict velocity_x,
    double *__restrict velocity_y,
    double *__restrict pressure
);

/**
 * \brief Convert the conserved variables to primitive variables.
 * 
 * \param system Pointer to the system struct.
 * 
 * \return Error status.
 */
ErrorStatus convert_conserved_to_primitive(System *__restrict system);


/**
 * \brief Convert the primitive variables to conserved variables.
 * 
 * \param system Pointer to the system struct.
 * 
 * \return Error status.
 */
ErrorStatus convert_primitive_to_conserved(System *__restrict system);

#endif
