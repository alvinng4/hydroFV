/**
 * \file system.c
 * 
 * \brief Functions related to the hydrodynamics system.
 * 
 * \author Ching-Yin Ng
 * \date 2025-03-08
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "common.h"
#include "error.h"
#include "system.h"

WIN32DLL_API System get_new_system_struct(void)
{
    System system = {
        .coord_sys = NULL,
        .boundary_condition_x_min = NULL,
        .boundary_condition_x_max = NULL,
        .boundary_condition_y_min = NULL,
        .boundary_condition_y_max = NULL,
        .boundary_condition_z_min = NULL,
        .boundary_condition_z_max = NULL,
        .gamma = -1.0,
        .x_min = -1.0,
        .x_max = -1.0,
        .y_min = -1.0,
        .y_max = -1.0,
        .z_min = -1.0,
        .z_max = -1.0,
        .num_cells_x = -1,
        .num_cells_y = -1,
        .num_cells_z = -1,
        .num_ghost_cells_side = -1,

        .coord_sys_flag_ = -1,
        .boundary_condition_flag_x_min_ = -1,
        .boundary_condition_flag_x_max_ = -1,
        .boundary_condition_flag_y_min_ = -1,
        .boundary_condition_flag_y_max_ = -1,
        .boundary_condition_flag_z_min_ = -1,
        .boundary_condition_flag_z_max_ = -1,
        .dx_ = -1.0,
        .dy_ = -1.0,
        .dz_ = -1.0,
        .density_ = NULL,
        .velocity_x_ = NULL,
        .velocity_y_ = NULL,
        .velocity_z_ = NULL,
        .pressure_ = NULL,
        .mass_ = NULL,
        .momentum_x_ = NULL,
        .momentum_y_ = NULL,
        .momentum_z_ = NULL,
        .energy_ = NULL,
        .mid_points_x_ = NULL,
        .mid_points_y_ = NULL,
        .mid_points_z_ = NULL,
        .volume_ = NULL
    };

    return system;
}

IN_FILE ErrorStatus check_init_system_input(const System *__restrict system)
{
    /* Coordinate system */
    if (!system->coord_sys)
    {
        return WRAP_RAISE_ERROR(VALUE_ERROR, "Coordinate system pointer is NULL.");
    }

    /* Gamma */
    if (system->gamma < 1.0)
    {
        size_t error_message_size = strlen("Gamma must be greater than 1. Got: ") + 1 + 128;
        char error_message[error_message_size];
        snprintf(error_message, error_message_size, "Gamma must be greater than 1. Got: %.3g", system->gamma);
        return WRAP_RAISE_ERROR(VALUE_ERROR, error_message);
    }

    /* Array pointers */
    if (system->density_)
    {
        return WRAP_RAISE_ERROR(VALUE_ERROR, "Density array pointer (system->density_) is not NULL.");
    }
    if (system->velocity_x_)
    {
        return WRAP_RAISE_ERROR(VALUE_ERROR, "Velocity (x-component) array pointer (system->velocity_x_) is not NULL.");
    }
    if (system->velocity_y_)
    {
        return WRAP_RAISE_ERROR(VALUE_ERROR, "Velocity (y-component) array pointer (system->velocity_y_) is not NULL.");
    }
    if (system->velocity_z_)
    {
        return WRAP_RAISE_ERROR(VALUE_ERROR, "Velocity (z-component) array pointer (system->velocity_z_) is not NULL.");
    }
    if (system->pressure_)
    {
        return WRAP_RAISE_ERROR(VALUE_ERROR, "Pressure array pointer (system->pressure_) is not NULL.");
    }
    if (system->mass_)
    {
        return WRAP_RAISE_ERROR(VALUE_ERROR, "Mass array pointer (system->mass_) is not NULL.");
    }
    if (system->momentum_x_)
    {
        return WRAP_RAISE_ERROR(VALUE_ERROR, "Momentum (x-component) array pointer (system->momentum_x_) is not NULL.");
    }
    if (system->momentum_y_)
    {
        return WRAP_RAISE_ERROR(VALUE_ERROR, "Momentum (y-component) array pointer (system->momentum_y_) is not NULL.");
    }
    if (system->momentum_z_)
    {
        return WRAP_RAISE_ERROR(VALUE_ERROR, "Momentum (z-component) array pointer (system->momentum_z_) is not NULL.");
    }
    if (system->energy_)
    {
        return WRAP_RAISE_ERROR(VALUE_ERROR, "Energy array pointer (system->energy_) is not NULL.");
    }
    if (system->mid_points_x_)
    {
        return WRAP_RAISE_ERROR(VALUE_ERROR, "Mid points x array pointer (system->mid_points_x_) is not NULL.");
    }
    if (system->mid_points_y_)
    {
        return WRAP_RAISE_ERROR(VALUE_ERROR, "Mid points y array pointer (system->mid_points_y_) is not NULL.");
    }
    if (system->mid_points_z_)
    {
        return WRAP_RAISE_ERROR(VALUE_ERROR, "Mid points z array pointer (system->mid_points_z_) is not NULL.");
    }
    if (system->volume_)
    {
        return WRAP_RAISE_ERROR(VALUE_ERROR, "Volume array pointer (system->volume_) is not NULL.");
    }

    /* Number of ghost cells */
    if (system->num_ghost_cells_side < 0)
    {
        size_t error_message_size = strlen("Number of ghost cells must be non-negative. Got: ") + 1 + 128;
        char error_message[error_message_size];
        snprintf(error_message, error_message_size, "Number of ghost cells must be non-negative. Got: %d", system->num_ghost_cells_side);
        return WRAP_RAISE_ERROR(VALUE_ERROR, error_message);
    }

    /* Number of cells, domain and boundary conditions */
    switch (system->coord_sys_flag_)
    {
        case COORD_SYS_CARTESIAN_3D:
            if (system->num_cells_z < 1)
            {
                size_t error_message_size = strlen("Number of cells in z-direction must be positive. Got: ") + 1 + 128;
                char error_message[error_message_size];
                snprintf(error_message, error_message_size, "Number of cells in z-direction must be positive. Got: %d", system->num_cells_z);
                return WRAP_RAISE_ERROR(VALUE_ERROR, error_message);
            }
            if (system->z_min >= system->z_max)
            {
                size_t error_message_size = strlen("z_min must be less than z_max. Got: z_min = , z_max = ") + 1 + 256;
                char error_message[error_message_size];
                snprintf(error_message, error_message_size, "z_min must be less than z_max. Got: z_min = %.3g, z_max = %.3g", system->z_min, system->z_max);
                return WRAP_RAISE_ERROR(VALUE_ERROR, error_message);
            }
            if (!system->boundary_condition_z_min)
            {
                return WRAP_RAISE_ERROR(VALUE_ERROR, "Boundary condition pointer for z_min is NULL.");
            }
            if (!system->boundary_condition_z_max)
            {
                return WRAP_RAISE_ERROR(VALUE_ERROR, "Boundary condition pointer for z_max is NULL.");
            }
            /* FALL THROUGH */
        case COORD_SYS_CARTESIAN_2D:
            if (system->num_cells_y < 1)
            {
                size_t error_message_size = strlen("Number of cells in y-direction must be positive. Got: ") + 1 + 128;
                char error_message[error_message_size];
                snprintf(error_message, error_message_size, "Number of cells in y-direction must be positive. Got: %d", system->num_cells_y);
                return WRAP_RAISE_ERROR(VALUE_ERROR, error_message);
            }
            if (system->y_min >= system->y_max)
            {
                size_t error_message_size = strlen("y_min must be less than y_max. Got: y_min = , y_max = ") + 1 + 256;
                char error_message[error_message_size];
                snprintf(error_message, error_message_size, "y_min must be less than y_max. Got: y_min = %.3g, y_max = %.3g", system->y_min, system->y_max);
                return WRAP_RAISE_ERROR(VALUE_ERROR, error_message);
            }
            if (!system->boundary_condition_y_min)
            {
                return WRAP_RAISE_ERROR(VALUE_ERROR, "Boundary condition pointer for y_min is NULL.");
            }
            if (!system->boundary_condition_y_max)
            {
                return WRAP_RAISE_ERROR(VALUE_ERROR, "Boundary condition pointer for y_max is NULL.");
            }
            /* FALL THROUGH */
        case COORD_SYS_CARTESIAN_1D: case COORD_SYS_CYLINDRICAL_1D: case COORD_SYS_SPHERICAL_1D:
            if (system->num_cells_x < 1)
            {
                size_t error_message_size = strlen("Number of cells in x-direction must be positive. Got: ") + 1 + 128;
                char error_message[error_message_size];
                snprintf(error_message, error_message_size, "Number of cells in x-direction must be positive. Got: %d", system->num_cells_x);
                return WRAP_RAISE_ERROR(VALUE_ERROR, error_message);
            }
            if (system->x_min >= system->x_max)
            {
                size_t error_message_size = strlen("x_min must be less than x_max. Got: x_min = , x_max = ") + 1 + 256;
                char error_message[error_message_size];
                snprintf(error_message, error_message_size, "x_min must be less than x_max. Got: x_min = %.3g, x_max = %.3g", system->x_min, system->x_max);
                return WRAP_RAISE_ERROR(VALUE_ERROR, error_message);
            }
            if (!system->boundary_condition_x_min)
            {
                return WRAP_RAISE_ERROR(VALUE_ERROR, "Boundary condition pointer for x_min is NULL.");
            }
            if (!system->boundary_condition_x_max)
            {
                return WRAP_RAISE_ERROR(VALUE_ERROR, "Boundary condition pointer for x_max is NULL.");
            }
            break;
        default:
            return WRAP_RAISE_ERROR(VALUE_ERROR, "Coordinate system flag not recognized.");
    }

    return make_success_error_status();
}

IN_FILE ErrorStatus get_coord_sys_flag(
    System *__restrict system
)
{
    const char *coord_sys = system->coord_sys;
    int *coord_sys_flag = &system->coord_sys_flag_;

    if (!coord_sys)
    {
        return WRAP_RAISE_ERROR(VALUE_ERROR, "Coordinate system pointer is NULL.");
    }
    if (!coord_sys_flag)
    {
        return WRAP_RAISE_ERROR(VALUE_ERROR, "Coordinate system flag pointer is NULL.");
    }

    if (strcmp(coord_sys, "cartesian_1d") == 0)
    {
        *coord_sys_flag = COORD_SYS_CARTESIAN_1D;
    }
    else if (strcmp(coord_sys, "cartesian_2d") == 0)
    {
        *coord_sys_flag = COORD_SYS_CARTESIAN_2D;
    }
    else if (strcmp(coord_sys, "cartesian_3d") == 0)
    {
        *coord_sys_flag = COORD_SYS_CARTESIAN_3D;
    }
    else if (strcmp(coord_sys, "cylindrical_1d") == 0)
    {
        *coord_sys_flag = COORD_SYS_CYLINDRICAL_1D;
    }
    else if (strcmp(coord_sys, "spherical_1d") == 0)
    {
        *coord_sys_flag = COORD_SYS_SPHERICAL_1D;
    }
    else
    {
        return WRAP_RAISE_ERROR(VALUE_ERROR, "Coordinate system not recognized.");
    }

    return make_success_error_status();
}

/**
 * \brief Set boundary condition flags based on the boundary condition strings in the system struct.
 * 
 * \param system Pointer to the system struct.
 * 
 * \retval SUCCESS if successful.
 * \retval ERROR_UNKNOWN_BOUNDARY_CONDITION_X_MIN if the boundary condition for x_min is not recognized.
 * \retval ERROR_UNKNOWN_BOUNDARY_CONDITION_X_MAX if the boundary condition for x_max is not recognized.
 * \retval ERROR_UNKNOWN_BOUNDARY_CONDITION_Y_MIN if the boundary condition for y_min is not recognized.
 * \retval ERROR_UNKNOWN_BOUNDARY_CONDITION_Y_MAX if the boundary condition for y_max is not recognized.
 * \retval ERROR_UNKNOWN_BOUNDARY_CONDITION_Z_MIN if the boundary condition for z_min is not recognized.
 * \retval ERROR_UNKNOWN_BOUNDARY_CONDITION_Z_MAX if the boundary condition for z_max is not recognized.
 * \retval ERROR_BOUNDARY_CONDITION_PERIODIC_MISMATCH_X if only one side of x_min and x_max is set to periodic boundary condition.
 * \retval ERROR_BOUNDARY_CONDITION_PERIODIC_MISMATCH_Y if only one side of y_min and y_max is set to periodic boundary condition.
 * \retval ERROR_BOUNDARY_CONDITION_PERIODIC_MISMATCH_Z if only one side of z_min and z_max is set to periodic boundary condition.
 */
IN_FILE ErrorStatus get_boundary_condition_flag(System *__restrict system)
{
    ErrorStatus error_status;

    const char* boundary_condition_x_min = system->boundary_condition_x_min;
    const char* boundary_condition_x_max = system->boundary_condition_x_max;
    const char* boundary_condition_y_min = system->boundary_condition_y_min;
    const char* boundary_condition_y_max = system->boundary_condition_y_max;
    const char* boundary_condition_z_min = system->boundary_condition_z_min;
    const char* boundary_condition_z_max = system->boundary_condition_z_max;

    switch (system->coord_sys_flag_)
    {
        /* z-direction */
        case COORD_SYS_CARTESIAN_3D:
            if (!boundary_condition_z_min || !boundary_condition_z_max)
            {
                error_status = WRAP_RAISE_ERROR(VALUE_ERROR, "Boundary condition for z_min or z_max is NULL.");
                goto err_null_boundary_condition;
            }

            if (strcmp(boundary_condition_z_min, "none") == 0)
            {
                system->boundary_condition_flag_z_min_ = BOUNDARY_CONDITION_NONE;
            }
            else if (strcmp(boundary_condition_z_min, "reflective") == 0)
            {
                system->boundary_condition_flag_z_min_ = BOUNDARY_CONDITION_REFLECTIVE;
            }
            else if (strcmp(boundary_condition_z_min, "transmissive") == 0)
            {
                system->boundary_condition_flag_z_min_ = BOUNDARY_CONDITION_TRANSMISSIVE;
            }
            else if (strcmp(boundary_condition_z_min, "periodic") == 0)
            {
                system->boundary_condition_flag_z_min_ = BOUNDARY_CONDITION_PERIODIC;
            }
            else
            {
                error_status = WRAP_RAISE_ERROR(VALUE_ERROR, "Boundary condition for z_min is not recognized.");
                goto err_unknown_boundary_condition;
            }

            if (strcmp(boundary_condition_z_max, "none") == 0)
            {
                system->boundary_condition_flag_z_max_ = BOUNDARY_CONDITION_NONE;
            }
            else if (strcmp(boundary_condition_z_max, "reflective") == 0)
            {
                system->boundary_condition_flag_z_max_ = BOUNDARY_CONDITION_REFLECTIVE;
            }
            else if (strcmp(boundary_condition_z_max, "transmissive") == 0)
            {
                system->boundary_condition_flag_z_max_ = BOUNDARY_CONDITION_TRANSMISSIVE;
            }
            else if (strcmp(boundary_condition_z_max, "periodic") == 0)
            {
                system->boundary_condition_flag_z_max_ = BOUNDARY_CONDITION_PERIODIC;
            }
            else
            {
                error_status = WRAP_RAISE_ERROR(VALUE_ERROR, "Boundary condition for z_max is not recognized.");
                goto err_unknown_boundary_condition;
            }

            /* FALL THROUGH */

        /* y-direction */
        case COORD_SYS_CARTESIAN_2D:
            if (!boundary_condition_y_min || !boundary_condition_y_max)
            {
                error_status = WRAP_RAISE_ERROR(VALUE_ERROR, "Boundary condition for y_min or y_max is NULL.");
                goto err_null_boundary_condition;
            }

            if (strcmp(boundary_condition_y_min, "none") == 0)
            {
                system->boundary_condition_flag_y_min_ = BOUNDARY_CONDITION_NONE;
            }
            else if (strcmp(boundary_condition_y_min, "reflective") == 0)
            {
                system->boundary_condition_flag_y_min_ = BOUNDARY_CONDITION_REFLECTIVE;
            }
            else if (strcmp(boundary_condition_y_min, "transmissive") == 0)
            {
                system->boundary_condition_flag_y_min_ = BOUNDARY_CONDITION_TRANSMISSIVE;
            }
            else if (strcmp(boundary_condition_y_min, "periodic") == 0)
            {
                system->boundary_condition_flag_y_min_ = BOUNDARY_CONDITION_PERIODIC;
            }
            else
            {
                error_status = WRAP_RAISE_ERROR(VALUE_ERROR, "Boundary condition for y_min is not recognized.");
                goto err_unknown_boundary_condition;
            }

            if (strcmp(boundary_condition_y_max, "none") == 0)
            {
                system->boundary_condition_flag_y_max_ = BOUNDARY_CONDITION_NONE;
            }
            else if (strcmp(boundary_condition_y_max, "reflective") == 0)
            {
                system->boundary_condition_flag_y_max_ = BOUNDARY_CONDITION_REFLECTIVE;
            }
            else if (strcmp(boundary_condition_y_max, "transmissive") == 0)
            {
                system->boundary_condition_flag_y_max_ = BOUNDARY_CONDITION_TRANSMISSIVE;
            }
            else if (strcmp(boundary_condition_y_max, "periodic") == 0)
            {
                system->boundary_condition_flag_y_max_ = BOUNDARY_CONDITION_PERIODIC;
            }
            else
            {
                error_status = WRAP_RAISE_ERROR(VALUE_ERROR, "Boundary condition for y_max is not recognized.");
                goto err_unknown_boundary_condition;
            }
        
            /* FALL THROUGH */

        /* x-direction */
        case COORD_SYS_CARTESIAN_1D: case COORD_SYS_CYLINDRICAL_1D: case COORD_SYS_SPHERICAL_1D:
            if (!boundary_condition_x_min || !boundary_condition_x_max)
            {
                error_status = WRAP_RAISE_ERROR(VALUE_ERROR, "Boundary condition for x_min or x_max is NULL.");
                goto err_null_boundary_condition;
            }

            if (strcmp(boundary_condition_x_min, "none") == 0)
            {
                system->boundary_condition_flag_x_min_ = BOUNDARY_CONDITION_NONE;
            }
            else if (strcmp(boundary_condition_x_min, "reflective") == 0)
            {
                system->boundary_condition_flag_x_min_ = BOUNDARY_CONDITION_REFLECTIVE;
            }
            else if (strcmp(boundary_condition_x_min, "transmissive") == 0)
            {
                system->boundary_condition_flag_x_min_ = BOUNDARY_CONDITION_TRANSMISSIVE;
            }
            else if (strcmp(boundary_condition_x_min, "periodic") == 0)
            {
                system->boundary_condition_flag_x_min_ = BOUNDARY_CONDITION_PERIODIC;
            }
            else
            {
                error_status = WRAP_RAISE_ERROR(VALUE_ERROR, "Boundary condition for x_min is not recognized.");
                goto err_unknown_boundary_condition;
            }

            if (strcmp(boundary_condition_x_max, "none") == 0)
            {
                system->boundary_condition_flag_x_max_ = BOUNDARY_CONDITION_NONE;
            }
            else if (strcmp(boundary_condition_x_max, "reflective") == 0)
            {
                system->boundary_condition_flag_x_max_ = BOUNDARY_CONDITION_REFLECTIVE;
            }
            else if (strcmp(boundary_condition_x_max, "transmissive") == 0)
            {
                system->boundary_condition_flag_x_max_ = BOUNDARY_CONDITION_TRANSMISSIVE;
            }
            else if (strcmp(boundary_condition_x_max, "periodic") == 0)
            {
                system->boundary_condition_flag_x_max_ = BOUNDARY_CONDITION_PERIODIC;
            }
            else
            {
                error_status = WRAP_RAISE_ERROR(VALUE_ERROR, "Boundary condition for x_max is not recognized.");
                goto err_unknown_boundary_condition;
            }

            break;
        
        default:
            error_status = WRAP_RAISE_ERROR(VALUE_ERROR, "Coordinate system flag not recognized.");
            goto err_unknown_coord_sys_flag;
    }

    /* Check for periodic boundary condition (both side must have the same BC) */
    switch (system->coord_sys_flag_)
    {
        case COORD_SYS_CARTESIAN_3D:
            if (
                system->boundary_condition_flag_z_min_ == BOUNDARY_CONDITION_PERIODIC
                || system->boundary_condition_flag_z_max_ == BOUNDARY_CONDITION_PERIODIC
            )
            {
                if (
                    system->boundary_condition_flag_z_min_ != system->boundary_condition_flag_z_max_
                )
                {
                    error_status = WRAP_RAISE_ERROR(VALUE_ERROR, "Periodic boundary condition mismatch for z_min and z_max.");
                    goto err_periodic_boundary_condition_mismatch;
                }
            }
            /* FALL THROUGH */

        case COORD_SYS_CARTESIAN_2D:
            if (
                system->boundary_condition_flag_y_min_ == BOUNDARY_CONDITION_PERIODIC
                || system->boundary_condition_flag_y_max_ == BOUNDARY_CONDITION_PERIODIC
            )
            {
                if (
                    system->boundary_condition_flag_y_min_ != system->boundary_condition_flag_y_max_
                )
                {
                    error_status = WRAP_RAISE_ERROR(VALUE_ERROR, "Periodic boundary condition mismatch for y_min and y_max.");
                    goto err_periodic_boundary_condition_mismatch;
                }
            }
            /* FALL THROUGH */

        case COORD_SYS_CARTESIAN_1D: case COORD_SYS_CYLINDRICAL_1D: case COORD_SYS_SPHERICAL_1D:
            if (
                system->boundary_condition_flag_x_min_ == BOUNDARY_CONDITION_PERIODIC
                || system->boundary_condition_flag_x_max_ == BOUNDARY_CONDITION_PERIODIC
            )
            {
                if (
                    system->boundary_condition_flag_x_min_ != system->boundary_condition_flag_x_max_
                )
                {
                    error_status = WRAP_RAISE_ERROR(VALUE_ERROR, "Periodic boundary condition mismatch for x_min and x_max.");
                    goto err_periodic_boundary_condition_mismatch;
                }
            }
            break;

        default:
            error_status = WRAP_RAISE_ERROR(VALUE_ERROR, "Coordinate system flag not recognized.");
            goto err_unknown_coord_sys_flag;
    }

    return make_success_error_status();

err_periodic_boundary_condition_mismatch:
err_unknown_coord_sys_flag:
err_unknown_boundary_condition:
err_null_boundary_condition:
    return error_status;
}

IN_FILE ErrorStatus initialize_cell_width(System *__restrict system)
{
    switch (system->coord_sys_flag_)
    {
        case COORD_SYS_CARTESIAN_3D:
            system->dz_ = (system->z_max - system->z_min) / (system->num_cells_z + 2 * system->num_ghost_cells_side);
            /* FALL THROUGH */
        case COORD_SYS_CARTESIAN_2D:
            system->dy_ = (system->y_max - system->y_min) / (system->num_cells_y + 2 * system->num_ghost_cells_side);
            /* FALL THROUGH */
        case COORD_SYS_CARTESIAN_1D: case COORD_SYS_CYLINDRICAL_1D: case COORD_SYS_SPHERICAL_1D:
            system->dx_ = (system->x_max - system->x_min) / (system->num_cells_x + 2 * system->num_ghost_cells_side);
            break;
        default:
            return WRAP_RAISE_ERROR(VALUE_ERROR, "Coordinate system flag not recognized.");
    }

    return make_success_error_status();
}

IN_FILE ErrorStatus initialize_mid_points(System *__restrict system)
{
    switch (system->coord_sys_flag_)
    {
        case COORD_SYS_CARTESIAN_3D:
        {
            const int total_num_cells_z = system->num_cells_z + 2 * system->num_ghost_cells_side;
            for (int i = 0; i < total_num_cells_z; i++)
            {
                system->mid_points_z_[i] = system->z_min + (i + 0.5) * system->dz_;
            }
        }
        /* FALL THROUGH */
        case COORD_SYS_CARTESIAN_2D:
        {
            const int total_num_cells_y = system->num_cells_y + 2 * system->num_ghost_cells_side;
            for (int i = 0; i < total_num_cells_y; i++)
            {
                system->mid_points_y_[i] = system->y_min + (i + 0.5) * system->dy_;
            }
        }
        /* FALL THROUGH */
        case COORD_SYS_CARTESIAN_1D: case COORD_SYS_CYLINDRICAL_1D: case COORD_SYS_SPHERICAL_1D:
        {
            const int total_num_cells_x = system->num_cells_x + 2 * system->num_ghost_cells_side;
            for (int i = 0; i < total_num_cells_x; i++)
            {
                system->mid_points_x_[i] = system->x_min + (i + 0.5) * system->dx_;
            }
            break;
        }
        default:
        {
            return WRAP_RAISE_ERROR(VALUE_ERROR, "Coordinate system flag not recognized.");
        }
    }

    return make_success_error_status();
}

IN_FILE ErrorStatus initialize_volume(System *__restrict system)
{
    switch (system->coord_sys_flag_)
    {
        case COORD_SYS_CARTESIAN_1D:
        {
            const int total_num_cells_x = system->num_cells_x + 2 * system->num_ghost_cells_side;
            const real volume = system->dx_;
            for (int i = 0; i < total_num_cells_x; i++)
            {
                system->volume_[i] = volume;
            }
            return make_success_error_status();
        }
        case COORD_SYS_CYLINDRICAL_1D: 
        {
            const int total_num_cells_x = system->num_cells_x + 2 * system->num_ghost_cells_side;
            const real dx = system->dx_;
            system->volume_[0] = M_PI * dx * dx;
            for (int i = 1; i < total_num_cells_x; i++)
            {
                const real r_min = system->x_min + (i - 1) * dx;
                const real r_max = system->x_min + i * dx;
                system->volume_[i] = M_PI * (r_max * r_max - r_min * r_min);
            }
            return make_success_error_status();
        }
        case COORD_SYS_SPHERICAL_1D:
        {
            const int total_num_cells_x = system->num_cells_x + 2 * system->num_ghost_cells_side;
            const real dx = system->dx_;
            system->volume_[0] = (4.0 / 3.0) * M_PI * dx * dx * dx;
            for (int i = 1; i < total_num_cells_x; i++)
            {
                const real r_min = system->x_min + (i - 1) * dx;
                const real r_max = system->x_min + i * dx;
                system->volume_[i] = (4.0 / 3.0) * M_PI * (r_max * r_max * r_max - r_min * r_min * r_min);
            }
            return make_success_error_status();
        }
        case COORD_SYS_CARTESIAN_2D:
        {
            const int total_num_cells_x = system->num_cells_x + 2 * system->num_ghost_cells_side;
            const int total_num_cells_y = system->num_cells_y + 2 * system->num_ghost_cells_side;
            const real volume = system->dx_ * system->dy_;
            for (int i = 0; i < total_num_cells_x; i++)
            {
                for (int j = 0; j < total_num_cells_y; j++)
                {
                    system->volume_[j * total_num_cells_x + i] = volume;
                }
            }
            return make_success_error_status();
        }
        case COORD_SYS_CARTESIAN_3D:
        {
            const int total_num_cells_x = system->num_cells_x + 2 * system->num_ghost_cells_side;
            const int total_num_cells_y = system->num_cells_y + 2 * system->num_ghost_cells_side;
            const int total_num_cells_z = system->num_cells_z + 2 * system->num_ghost_cells_side;
            const real volume = system->dx_ * system->dy_ * system->dz_;
            for (int i = 0; i < total_num_cells_x; i++)
            {
                for (int j = 0; j < total_num_cells_y; j++)
                {
                    for (int k = 0; k < total_num_cells_z; k++)
                    {
                        system->volume_[
                            k * total_num_cells_x * total_num_cells_y + j * total_num_cells_y + i
                        ] = volume;
                    }
                }
            }
            return make_success_error_status();
        }
        default:
        {
            return WRAP_RAISE_ERROR(VALUE_ERROR, "Coordinate system flag not recognized.");
        }
    }
}

ErrorStatus system_init(System *__restrict system)
{
    ErrorStatus error_status;

    /* Get coordinate system flag */
    error_status = WRAP_TRACEBACK(get_coord_sys_flag(system));
    if (error_status.return_code != SUCCESS)
    {
        goto err_coord_sys_flag;
    }

    /* Check input (must be done after getting coord_sys_flag) */
    error_status = WRAP_TRACEBACK(check_init_system_input(system));
    if (error_status.return_code != SUCCESS)
    {
        goto err_init_input;
    }

    /* Get boundary condition flags */
    error_status = WRAP_TRACEBACK(get_boundary_condition_flag(system));
    if (error_status.return_code != SUCCESS)
    {
        goto err_boundary_condition_flag;
    }

    /* Calculate total number of cells */
    int total_num_cells;

    switch (system->coord_sys_flag_)
    {
        case COORD_SYS_CARTESIAN_1D: case COORD_SYS_CYLINDRICAL_1D: case COORD_SYS_SPHERICAL_1D:
            total_num_cells = system->num_cells_x + 2 * system->num_ghost_cells_side;
            break;
        case COORD_SYS_CARTESIAN_2D:
            total_num_cells = (
                system->num_cells_x + 2 * system->num_ghost_cells_side
            ) * (
                system->num_cells_y + 2 * system->num_ghost_cells_side
            );
            break;
        case COORD_SYS_CARTESIAN_3D:
            total_num_cells = (
                system->num_cells_x + 2 * system->num_ghost_cells_side
            ) * (
                system->num_cells_y + 2 * system->num_ghost_cells_side
            ) * (
                system->num_cells_z + 2 * system->num_ghost_cells_side
            );
            break;
        default:
            error_status = WRAP_RAISE_ERROR(VALUE_ERROR, "Coordinate system flag not recognized.");
            goto err_unknown_coord_sys_flag_total_num_cells;
    }

    /* Allocate memory */
    system->density_ = calloc(total_num_cells, sizeof(real));
    system->pressure_ = calloc(total_num_cells, sizeof(real));
    system->mass_ = calloc(total_num_cells, sizeof(real));
    system->energy_ = calloc(total_num_cells, sizeof(real));
    system->volume_ = malloc(total_num_cells * sizeof(real));

    if (
        !system->density_
        || !system->pressure_
        || !system->mass_
        || !system->energy_
        || !system->volume_
    )
    {
        error_status = WRAP_RAISE_ERROR(MEMORY_ERROR, "Memory allocation failed.");
        goto err_init_memory_alloc;
    }

    switch (system->coord_sys_flag_)
    {
        case COORD_SYS_CARTESIAN_3D:
        {
            const int total_num_cells_z = system->num_cells_z + 2 * system->num_ghost_cells_side;
            system->mid_points_z_ = malloc(total_num_cells_z * sizeof(real));
            system->velocity_z_ = calloc(total_num_cells, sizeof(real));
            system->momentum_z_ = calloc(total_num_cells, sizeof(real));
            if (!system->mid_points_z_ || !system->velocity_z_ || !system->momentum_z_)
            {
                error_status = WRAP_RAISE_ERROR(MEMORY_ERROR, "Memory allocation failed.");
                goto err_init_memory_alloc_z;
            }
        }
            /* FALL THROUGH */
        case COORD_SYS_CARTESIAN_2D:
        {
            const int total_num_cells_y = system->num_cells_y + 2 * system->num_ghost_cells_side;
            system->mid_points_y_ = malloc(total_num_cells_y * sizeof(real));
            system->velocity_y_ = calloc(total_num_cells, sizeof(real));
            system->momentum_y_ = calloc(total_num_cells, sizeof(real));
            if (!system->mid_points_y_ || !system->velocity_y_ || !system->momentum_y_)
            {
                error_status = WRAP_RAISE_ERROR(MEMORY_ERROR, "Memory allocation failed.");
                goto err_init_memory_alloc_y;
            }
        }
            /* FALL THROUGH */
        case COORD_SYS_CARTESIAN_1D: case COORD_SYS_CYLINDRICAL_1D: case COORD_SYS_SPHERICAL_1D:
        {
            const int total_num_cells_x = system->num_cells_x + 2 * system->num_ghost_cells_side;
            system->mid_points_x_ = malloc(total_num_cells_x * sizeof(real));
            system->velocity_x_ = calloc(total_num_cells, sizeof(real));
            system->momentum_x_ = calloc(total_num_cells, sizeof(real));
            if (!system->mid_points_x_ || !system->velocity_x_ || !system->momentum_x_)
            {
                error_status = WRAP_RAISE_ERROR(MEMORY_ERROR, "Memory allocation failed.");
                goto err_init_memory_alloc_x;
            }
        }
            break;
        default:
            error_status = WRAP_RAISE_ERROR(VALUE_ERROR, "Coordinate system flag not recognized.");
            goto err_unknown_coord_sys_flag_mid_points_malloc;
    }

    /* Initialize system attributes */
    error_status = WRAP_TRACEBACK(initialize_cell_width(system));
    if (error_status.return_code != SUCCESS)
    {
        goto err_init_sys_attr;
    }
    error_status = WRAP_TRACEBACK(initialize_mid_points(system));
    if (error_status.return_code != SUCCESS)
    {
        goto err_init_sys_attr;
    }
    error_status = WRAP_TRACEBACK(initialize_volume(system));
    if (error_status.return_code != SUCCESS)
    {
        goto err_init_sys_attr;
    }

    return make_success_error_status();

err_init_sys_attr:
err_init_memory_alloc_z:
    free(system->mid_points_z_);
    free(system->velocity_z_);
    free(system->momentum_z_);
err_init_memory_alloc_y:
    free(system->mid_points_y_);
    free(system->velocity_y_);
    free(system->momentum_y_);
err_init_memory_alloc_x:
    free(system->mid_points_x_);
    free(system->velocity_x_);
    free(system->momentum_x_);
err_unknown_coord_sys_flag_mid_points_malloc:
err_init_memory_alloc:
    free(system->density_);
    free(system->pressure_);
    free(system->mass_);
    free(system->energy_);
    free(system->mid_points_x_);
    free(system->volume_);
err_unknown_coord_sys_flag_total_num_cells:
err_boundary_condition_flag:
err_init_input:
err_coord_sys_flag:
    return error_status;
}

void free_system_memory(System *__restrict system)
{
    free(system->density_);
    free(system->velocity_x_);
    free(system->velocity_y_);
    free(system->velocity_z_);
    free(system->pressure_);
    free(system->mass_);
    free(system->momentum_x_);
    free(system->momentum_y_);
    free(system->momentum_z_);
    free(system->energy_);
    free(system->mid_points_x_);
    free(system->mid_points_y_);
    free(system->mid_points_z_);
    free(system->volume_);
}

IN_FILE ErrorStatus set_boundary_condition_1d(System *__restrict system)
{
    ErrorStatus error_status;

    real *__restrict density = system->density_;
    real *__restrict velocity_x = system->velocity_x_;
    real *__restrict pressure = system->pressure_;
    const int num_ghost_cells_side = system->num_ghost_cells_side;

    switch (system->boundary_condition_flag_x_min_)
    {
        case BOUNDARY_CONDITION_NONE:
            break;
        case BOUNDARY_CONDITION_REFLECTIVE:
            for (int i = 0; i < num_ghost_cells_side; i++)
            {
                density[num_ghost_cells_side - 1 - i] = density[num_ghost_cells_side + i];
                velocity_x[num_ghost_cells_side - 1 - i] = -velocity_x[num_ghost_cells_side + i];
                pressure[num_ghost_cells_side - 1 - i] = pressure[num_ghost_cells_side + i];
            }
            break;
        case BOUNDARY_CONDITION_TRANSMISSIVE:
            for (int i = 0; i < num_ghost_cells_side; i++)
            {
                density[num_ghost_cells_side - 1 - i] = density[num_ghost_cells_side + i];
                velocity_x[num_ghost_cells_side - 1 - i] = velocity_x[num_ghost_cells_side + i];
                pressure[num_ghost_cells_side - 1 - i] = pressure[num_ghost_cells_side + i];
            }
            break;
        case BOUNDARY_CONDITION_PERIODIC:
            /* To be processed together with in x_max */
            break;
        default:
            error_status = WRAP_RAISE_ERROR(VALUE_ERROR, "Boundary condition flag not recognized for x_min.");
            goto err_unknown_boundary_condition_flag;
    }

    switch (system->boundary_condition_flag_x_max_)
    {
        case BOUNDARY_CONDITION_NONE:
            break;
        case BOUNDARY_CONDITION_REFLECTIVE:
            for (int i = 0; i < num_ghost_cells_side; i++)
            {
                density[num_ghost_cells_side + system->num_cells_x + i] = density[num_ghost_cells_side + system->num_cells_x - 1 - i];
                velocity_x[num_ghost_cells_side + system->num_cells_x + i] = -velocity_x[num_ghost_cells_side + system->num_cells_x - 1 - i];
                pressure[num_ghost_cells_side + system->num_cells_x + i] = pressure[num_ghost_cells_side + system->num_cells_x - 1 - i];
            }
            break;
        case BOUNDARY_CONDITION_TRANSMISSIVE:
            for (int i = 0; i < num_ghost_cells_side; i++)
            {
                density[num_ghost_cells_side + system->num_cells_x + i] = density[num_ghost_cells_side + system->num_cells_x - 1 - i];
                velocity_x[num_ghost_cells_side + system->num_cells_x + i] = velocity_x[num_ghost_cells_side + system->num_cells_x - 1 - i];
                pressure[num_ghost_cells_side + system->num_cells_x + i] = pressure[num_ghost_cells_side + system->num_cells_x - 1 - i];
            }
            break;
        case BOUNDARY_CONDITION_PERIODIC:
            for (int i = 0; i < num_ghost_cells_side; i++)
            {
                density[num_ghost_cells_side + system->num_cells_x + i] = density[num_ghost_cells_side + i];
                velocity_x[num_ghost_cells_side + system->num_cells_x + i] = velocity_x[num_ghost_cells_side + i];
                pressure[num_ghost_cells_side + system->num_cells_x + i] = pressure[num_ghost_cells_side + i];

                density[num_ghost_cells_side - 1 - i] = density[num_ghost_cells_side + system->num_cells_x - 1 - i];
                velocity_x[num_ghost_cells_side - 1 - i] = velocity_x[num_ghost_cells_side + system->num_cells_x - 1 - i];
                pressure[num_ghost_cells_side - 1 - i] = pressure[num_ghost_cells_side + system->num_cells_x - 1 - i];
            }
            break;
        default:
            error_status = WRAP_RAISE_ERROR(VALUE_ERROR, "Boundary condition flag not recognized for x_max.");
            goto err_unknown_boundary_condition_flag;
    }

    return make_success_error_status();

err_unknown_boundary_condition_flag:
    return error_status;
}

IN_FILE ErrorStatus set_boundary_condition_cartesian_2d(System *__restrict system)
{
    ErrorStatus error_status;

    real *__restrict density = system->density_;
    real *__restrict velocity_x = system->velocity_x_;
    real *__restrict velocity_y = system->velocity_y_;
    real *__restrict pressure = system->pressure_;
    const int num_ghost_cells_side = system->num_ghost_cells_side;
    const int num_cells_x = system->num_cells_x;
    const int num_cells_y = system->num_cells_y;
    const int total_num_cells_x = num_cells_x + 2 * num_ghost_cells_side;

    switch (system->boundary_condition_flag_x_min_)
    {
        case BOUNDARY_CONDITION_NONE:
            break;
        case BOUNDARY_CONDITION_REFLECTIVE:
            for (int i = 0; i < num_ghost_cells_side; i++)
            {
                for (int j = num_ghost_cells_side; j < (num_ghost_cells_side + num_cells_y); j++)
                {
                    density[j * total_num_cells_x + (num_ghost_cells_side - 1 - i)] = density[j * total_num_cells_x + (num_ghost_cells_side + i)];
                    velocity_x[j * total_num_cells_x + (num_ghost_cells_side - 1 - i)] = -velocity_x[j * total_num_cells_x + (num_ghost_cells_side + i)];
                    velocity_y[j * total_num_cells_x + (num_ghost_cells_side - 1 - i)] = velocity_y[j * total_num_cells_x + (num_ghost_cells_side + i)];
                    pressure[j * total_num_cells_x + (num_ghost_cells_side - 1 - i)] = pressure[j * total_num_cells_x + (num_ghost_cells_side + i)];
                }
            }
            break;
        case BOUNDARY_CONDITION_TRANSMISSIVE:
            for (int i = 0; i < num_ghost_cells_side; i++)
            {
                for (int j = num_ghost_cells_side; j < (num_ghost_cells_side + num_cells_y); j++)
                {
                    density[j * total_num_cells_x + (num_ghost_cells_side - 1 - i)] = density[j * total_num_cells_x + (num_ghost_cells_side + i)];
                    velocity_x[j * total_num_cells_x + (num_ghost_cells_side - 1 - i)] = velocity_x[j * total_num_cells_x + (num_ghost_cells_side + i)];
                    velocity_y[j * total_num_cells_x + (num_ghost_cells_side - 1 - i)] = velocity_y[j * total_num_cells_x + (num_ghost_cells_side + i)];
                    pressure[j * total_num_cells_x + (num_ghost_cells_side - 1 - i)] = pressure[j * total_num_cells_x + (num_ghost_cells_side + i)];
                }
            }
            break;
        case BOUNDARY_CONDITION_PERIODIC:
            /* To be processed together with in x_max */
            break;
        default:
            error_status = WRAP_RAISE_ERROR(VALUE_ERROR, "Boundary condition flag not recognized for x_min.");
            goto err_unknown_boundary_condition_flag;
    }

    switch (system->boundary_condition_flag_x_max_)
    {
        case BOUNDARY_CONDITION_NONE:
            break;
        case BOUNDARY_CONDITION_REFLECTIVE:
            for (int i = 0; i < num_ghost_cells_side; i++)
            {
                for (int j = num_ghost_cells_side; j < (num_ghost_cells_side + num_cells_y); j++)
                {
                    density[j * total_num_cells_x + (num_ghost_cells_side + num_cells_x + i)] = density[j * total_num_cells_x + (num_ghost_cells_side + num_cells_x - 1 - i)];
                    velocity_x[j * total_num_cells_x + (num_ghost_cells_side + num_cells_x + i)] = -velocity_x[j * total_num_cells_x + (num_ghost_cells_side + num_cells_x - 1 - i)];
                    velocity_y[j * total_num_cells_x + (num_ghost_cells_side + num_cells_x + i)] = velocity_y[j * total_num_cells_x + (num_ghost_cells_side + num_cells_x - 1 - i)];
                    pressure[j * total_num_cells_x + (num_ghost_cells_side + num_cells_x + i)] = pressure[j * total_num_cells_x + (num_ghost_cells_side + num_cells_x - 1 - i)];
                }
            }
            break;
        case BOUNDARY_CONDITION_TRANSMISSIVE:
            for (int i = 0; i < num_ghost_cells_side; i++)
            {
                for (int j = num_ghost_cells_side; j < (num_ghost_cells_side + num_cells_y); j++)
                {
                    density[j * total_num_cells_x + (num_ghost_cells_side + num_cells_x + i)] = density[j * total_num_cells_x + (num_ghost_cells_side + num_cells_x - 1 - i)];
                    velocity_x[j * total_num_cells_x + (num_ghost_cells_side + num_cells_x + i)] = velocity_x[j * total_num_cells_x + (num_ghost_cells_side + num_cells_x - 1 - i)];
                    velocity_y[j * total_num_cells_x + (num_ghost_cells_side + num_cells_x + i)] = velocity_y[j * total_num_cells_x + (num_ghost_cells_side + num_cells_x - 1 - i)];
                    pressure[j * total_num_cells_x + (num_ghost_cells_side + num_cells_x + i)] = pressure[j * total_num_cells_x + (num_ghost_cells_side + num_cells_x - 1 - i)];
                }
            }
            break;
        case BOUNDARY_CONDITION_PERIODIC:
            for (int i = 0; i < num_ghost_cells_side; i++)
            {
                for (int j = num_ghost_cells_side; j < (num_ghost_cells_side + num_cells_y); j++)
                {
                    density[j * total_num_cells_x + (num_ghost_cells_side + num_cells_x + i)] = density[j * total_num_cells_x + (num_ghost_cells_side + i)];
                    velocity_x[j * total_num_cells_x + (num_ghost_cells_side + num_cells_x + i)] = velocity_x[j * total_num_cells_x + (num_ghost_cells_side + i)];
                    velocity_y[j * total_num_cells_x + (num_ghost_cells_side + num_cells_x + i)] = velocity_y[j * total_num_cells_x + (num_ghost_cells_side + i)];
                    pressure[j * total_num_cells_x + (num_ghost_cells_side + num_cells_x + i)] = pressure[j * total_num_cells_x + (num_ghost_cells_side + i)];

                    density[j * total_num_cells_x + (num_ghost_cells_side - 1 - i)] = density[j * total_num_cells_x + (num_ghost_cells_side + num_cells_x - 1 - i)];
                    velocity_x[j * total_num_cells_x + (num_ghost_cells_side - 1 - i)] = velocity_x[j * total_num_cells_x + (num_ghost_cells_side + num_cells_x - 1 - i)];
                    velocity_y[j * total_num_cells_x + (num_ghost_cells_side - 1 - i)] = velocity_y[j * total_num_cells_x + (num_ghost_cells_side + num_cells_x - 1 - i)];
                    pressure[j * total_num_cells_x + (num_ghost_cells_side - 1 - i)] = pressure[j * total_num_cells_x + (num_ghost_cells_side + num_cells_x - 1 - i)];
                }
            }
            break;
        default:
            error_status = WRAP_RAISE_ERROR(VALUE_ERROR, "Boundary condition flag not recognized for x_max.");
            goto err_unknown_boundary_condition_flag;
    }

    switch (system->boundary_condition_flag_y_min_)
    {
        case BOUNDARY_CONDITION_NONE:
            break;
        case BOUNDARY_CONDITION_REFLECTIVE:
            for (int i = num_ghost_cells_side; i < (num_ghost_cells_side + num_cells_x); i++)
            {
                for (int j = 0; j < num_ghost_cells_side; j++)
                {
                    density[(num_ghost_cells_side - 1 - j) * total_num_cells_x + i] = density[(num_ghost_cells_side + j) * total_num_cells_x + i];
                    velocity_x[(num_ghost_cells_side - 1 - j) * total_num_cells_x + i] = velocity_x[(num_ghost_cells_side + j) * total_num_cells_x + i];
                    velocity_y[(num_ghost_cells_side - 1 - j) * total_num_cells_x + i] = -velocity_y[(num_ghost_cells_side + j) * total_num_cells_x + i];
                    pressure[(num_ghost_cells_side - 1 - j) * total_num_cells_x + i] = pressure[(num_ghost_cells_side + j) * total_num_cells_x + i];
                }
            }
            break;
        case BOUNDARY_CONDITION_TRANSMISSIVE:
            for (int i = num_ghost_cells_side; i < (num_ghost_cells_side + num_cells_x); i++)
            {
                for (int j = 0; j < num_ghost_cells_side; j++)
                {
                    density[(num_ghost_cells_side - 1 - j) * total_num_cells_x + i] = density[(num_ghost_cells_side + j) * total_num_cells_x + i];
                    velocity_x[(num_ghost_cells_side - 1 - j) * total_num_cells_x + i] = velocity_x[(num_ghost_cells_side + j) * total_num_cells_x + i];
                    velocity_y[(num_ghost_cells_side - 1 - j) * total_num_cells_x + i] = velocity_y[(num_ghost_cells_side + j) * total_num_cells_x + i];
                    pressure[(num_ghost_cells_side - 1 - j) * total_num_cells_x + i] = pressure[(num_ghost_cells_side + j) * total_num_cells_x + i];
                }
            }
            break;
        case BOUNDARY_CONDITION_PERIODIC:
            /* To be processed together with in y_max */
            break;
        default:
            error_status = WRAP_RAISE_ERROR(VALUE_ERROR, "Boundary condition flag not recognized for y_min.");
            goto err_unknown_boundary_condition_flag;
    }

    switch (system->boundary_condition_flag_y_max_)
    {
        case BOUNDARY_CONDITION_NONE:
            break;
        case BOUNDARY_CONDITION_REFLECTIVE:
            for (int i = num_ghost_cells_side; i < (num_ghost_cells_side + num_cells_x); i++)
            {
                for (int j = 0; j < num_ghost_cells_side; j++)
                {
                    density[(num_ghost_cells_side + num_cells_y + j) * total_num_cells_x + i] = density[(num_ghost_cells_side + num_cells_y - 1 - j) * total_num_cells_x + i];
                    velocity_x[(num_ghost_cells_side + num_cells_y + j) * total_num_cells_x + i] = velocity_x[(num_ghost_cells_side + num_cells_y - 1 - j) * total_num_cells_x + i];
                    velocity_y[(num_ghost_cells_side + num_cells_y + j) * total_num_cells_x + i] = -velocity_y[(num_ghost_cells_side + num_cells_y - 1 - j) * total_num_cells_x + i];
                    pressure[(num_ghost_cells_side + num_cells_y + j) * total_num_cells_x + i] = pressure[(num_ghost_cells_side + num_cells_y - 1 - j) * total_num_cells_x + i];
                }
            }
            break;
        case BOUNDARY_CONDITION_TRANSMISSIVE:
            for (int i = num_ghost_cells_side; i < (num_ghost_cells_side + num_cells_x); i++)
            {
                for (int j = 0; j < num_ghost_cells_side; j++)
                {
                    density[(num_ghost_cells_side + num_cells_y + j) * total_num_cells_x + i] = density[(num_ghost_cells_side + num_cells_y - 1 - j) * total_num_cells_x + i];
                    velocity_x[(num_ghost_cells_side + num_cells_y + j) * total_num_cells_x + i] = velocity_x[(num_ghost_cells_side + num_cells_y - 1 - j) * total_num_cells_x + i];
                    velocity_y[(num_ghost_cells_side + num_cells_y + j) * total_num_cells_x + i] = velocity_y[(num_ghost_cells_side + num_cells_y - 1 - j) * total_num_cells_x + i];
                    pressure[(num_ghost_cells_side + num_cells_y + j) * total_num_cells_x + i] = pressure[(num_ghost_cells_side + num_cells_y - 1 - j) * total_num_cells_x + i];
                }
            }
            break;
        case BOUNDARY_CONDITION_PERIODIC:
            for (int i = num_ghost_cells_side; i < (num_ghost_cells_side + num_cells_x); i++)
            {
                for (int j = 0; j < num_ghost_cells_side; j++)
                {
                    density[(num_ghost_cells_side + num_cells_y + j) * total_num_cells_x + i] = density[(num_ghost_cells_side + j) * total_num_cells_x + i];
                    velocity_x[(num_ghost_cells_side + num_cells_y + j) * total_num_cells_x + i] = velocity_x[(num_ghost_cells_side + j) * total_num_cells_x + i];
                    velocity_y[(num_ghost_cells_side + num_cells_y + j) * total_num_cells_x + i] = velocity_y[(num_ghost_cells_side + j) * total_num_cells_x + i];
                    pressure[(num_ghost_cells_side + num_cells_y + j) * total_num_cells_x + i] = pressure[(num_ghost_cells_side + j) * total_num_cells_x + i];

                    density[(num_ghost_cells_side - 1 - j) * total_num_cells_x + i] = density[(num_ghost_cells_side + num_cells_y - 1 - j) * total_num_cells_x + i];
                    velocity_x[(num_ghost_cells_side - 1 - j) * total_num_cells_x + i] = velocity_x[(num_ghost_cells_side + num_cells_y - 1 - j) * total_num_cells_x + i];
                    velocity_y[(num_ghost_cells_side - 1 - j) * total_num_cells_x + i] = velocity_y[(num_ghost_cells_side + num_cells_y - 1 - j) * total_num_cells_x + i];
                    pressure[(num_ghost_cells_side - 1 - j) * total_num_cells_x + i] = pressure[(num_ghost_cells_side + num_cells_y - 1 - j) * total_num_cells_x + i];
                }
            }
            break;
        default:
            error_status = WRAP_RAISE_ERROR(VALUE_ERROR, "Boundary condition flag not recognized for y_max.");
            goto err_unknown_boundary_condition_flag;
    }

    return make_success_error_status();

err_unknown_boundary_condition_flag:
    return error_status;
}

IN_FILE ErrorStatus set_boundary_condition_cartesian_3d(System *__restrict system)
{
    (void) system;
    return WRAP_RAISE_ERROR(NOT_IMPLEMENTED_ERROR, "");
}

ErrorStatus set_boundary_condition(System *__restrict system)
{
    switch (system->coord_sys_flag_)
    {
        case COORD_SYS_CARTESIAN_1D: case COORD_SYS_CYLINDRICAL_1D: case COORD_SYS_SPHERICAL_1D:
            return WRAP_TRACEBACK(set_boundary_condition_1d(system));
        case COORD_SYS_CARTESIAN_2D:
            return WRAP_TRACEBACK(set_boundary_condition_cartesian_2d(system));
        case COORD_SYS_CARTESIAN_3D:
            return WRAP_TRACEBACK(set_boundary_condition_cartesian_3d(system));
        default:
            return WRAP_RAISE_ERROR(VALUE_ERROR, "Coordinate system flag not recognized.");
    }
}

ErrorStatus convert_conserved_to_primitive(System *__restrict system)
{
    const int num_ghost_cells_side = system->num_ghost_cells_side;
    const real gamma = system->gamma;
    const real *__restrict volume = system->volume_;
    const real *__restrict mass = system->mass_;
    const real *__restrict momentum_x = system->momentum_x_;
    const real *__restrict momentum_y = system->momentum_y_;
    // const real *__restrict momentum_z = system->momentum_z_;
    const real *__restrict energy = system->energy_;
    real *__restrict density = system->density_;
    real *__restrict velocity_x = system->velocity_x_;
    real *__restrict velocity_y = system->velocity_y_;
    // real *__restrict velocity_z = system->velocity_z_;
    real *__restrict pressure = system->pressure_;

    switch(system->coord_sys_flag_)
    {
        case COORD_SYS_CARTESIAN_1D: case COORD_SYS_CYLINDRICAL_1D: case COORD_SYS_SPHERICAL_1D:
            for (int i = num_ghost_cells_side; i < (system->num_cells_x + num_ghost_cells_side); i++)
            {
                const real mass_i = mass[i];
                const real momentum_x_i = momentum_x[i];
                const real energy_i = energy[i];
                const real volume_i = volume[i];
                density[i] = mass_i / volume_i;
                velocity_x[i] = momentum_x_i / mass_i;
                pressure[i] = (gamma - 1.0) * (energy_i - 0.5 * mass_i * velocity_x[i] * velocity_x[i]) / volume_i;
            }
            break;
        case COORD_SYS_CARTESIAN_2D:
            for (int i = num_ghost_cells_side; i < (system->num_cells_x + num_ghost_cells_side); i++)
            {
                for (int j = num_ghost_cells_side; j < (system->num_cells_y + num_ghost_cells_side); j++)
                {
                    const int index = j * (system->num_cells_x + 2 * num_ghost_cells_side) + i;
                    const real mass_ij = mass[index];
                    const real momentum_x_ij = momentum_x[index];
                    const real momentum_y_ij = momentum_y[index];
                    const real energy_ij = energy[index];
                    const real volume_ij = volume[index];
                    density[index] = mass_ij / volume_ij;
                    velocity_x[index] = momentum_x_ij / mass_ij;
                    velocity_y[index] = momentum_y_ij / mass_ij;
                    pressure[index] = (gamma - 1.0) * (
                        energy_ij - 0.5 * mass_ij * (
                            velocity_x[index] * velocity_x[index] + velocity_y[index] * velocity_y[index]
                        )
                    ) / volume_ij;
                }
            }
            break;
        case COORD_SYS_CARTESIAN_3D:
            return WRAP_RAISE_ERROR(NOT_IMPLEMENTED_ERROR, "");
            break;
        default:
            return WRAP_RAISE_ERROR(VALUE_ERROR, "Coordinate system flag not recognized.");
    }

    return make_success_error_status();
}

ErrorStatus convert_primitive_to_conserved(System *__restrict system)
{
    const int num_ghost_cells_side = system->num_ghost_cells_side;
    const real gamma = system->gamma;
    const real *__restrict volume = system->volume_;
    const real *__restrict density = system->density_;
    const real *__restrict velocity_x = system->velocity_x_;
    const real *__restrict velocity_y = system->velocity_y_;
    // const real *__restrict velocity_z = system->velocity_z_;
    const real *__restrict pressure = system->pressure_;
    real *__restrict mass = system->mass_;
    real *__restrict momentum_x = system->momentum_x_;
    real *__restrict momentum_y = system->momentum_y_;
    // real *__restrict momentum_z = system->momentum_z_;
    real *__restrict energy = system->energy_;

    switch(system->coord_sys_flag_)
    {
        case COORD_SYS_CARTESIAN_1D: case COORD_SYS_CYLINDRICAL_1D: case COORD_SYS_SPHERICAL_1D:
            for (int i = num_ghost_cells_side; i < (system->num_cells_x + num_ghost_cells_side); i++)
            {
                const real density_i = density[i];
                const real velocity_x_i = velocity_x[i];
                const real pressure_i = pressure[i];
                const real volume_i = volume[i];
                mass[i] = density_i * volume_i;
                momentum_x[i] = mass[i] * velocity_x_i;
                energy[i] = volume[i] * (
                    0.5 * density_i * velocity_x_i * velocity_x_i + pressure_i / (gamma - 1.0)
                );
            }
            break;
        case COORD_SYS_CARTESIAN_2D:
            for (int i = num_ghost_cells_side; i < (system->num_cells_x + num_ghost_cells_side); i++)
            {
                for (int j = num_ghost_cells_side; j < (system->num_cells_y + num_ghost_cells_side); j++)
                {
                    const int index = j * (system->num_cells_x + 2 * num_ghost_cells_side) + i;
                    const real density_ij = density[index];
                    const real velocity_x_ij = velocity_x[index];
                    const real velocity_y_ij = velocity_y[index];
                    const real pressure_ij = pressure[index];
                    const real volume_ij = volume[index];
                    mass[index] = density_ij * volume_ij;
                    momentum_x[index] = mass[index] * velocity_x_ij;
                    momentum_y[index] = mass[index] * velocity_y_ij;
                    energy[index] = volume_ij * (
                        0.5 * density_ij * (
                            velocity_x_ij * velocity_x_ij + velocity_y_ij * velocity_y_ij
                        ) + pressure_ij / (gamma - 1.0)
                    );
                }
            }
            break;
        case COORD_SYS_CARTESIAN_3D:
            return WRAP_RAISE_ERROR(NOT_IMPLEMENTED_ERROR, "");
            break;
        default:
            return WRAP_RAISE_ERROR(VALUE_ERROR, "Coordinate system flag not recognized.");
    }

    return make_success_error_status();
}
