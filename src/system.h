/**
 * \file system.h
 * 
 * \brief Header file for definition and prototypes related to the hydrodynamics system.
 * 
 * \author Ching-Yin Ng
 * \date 2025-03-21
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

typedef struct System
{
    const char *coord_sys;
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
