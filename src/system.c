/**
 * \file system.c
 * 
 * \brief Functions related to the hydrodynamics system.
 * 
 * \author Ching-Yin Ng
 * \date 2025-03-21
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef USE_OPENMP
#include <omp.h>
#endif

#include "common.h"
#include "error.h"
#include "system.h"

System get_new_system_struct(void)
{
    System system = {
        .coord_sys = NULL,
        .gamma = -1.0,
        .gravity = 0.0,
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
        .surface_area_x_ = NULL,
        .surface_area_y_ = NULL,
        .surface_area_z_ = NULL,
        .volume_ = NULL
    };

    return system;
}

/**
 * \brief Check the system input for the system_init function.
 * 
 * \param system Pointer to the system struct.
 * 
 * \return Error status.
 */
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
        const size_t error_message_size = strlen("Gamma must be greater than 1. Got: ") + 1 + 128;
        char error_message[error_message_size];
        snprintf(error_message, error_message_size, "Gamma must be greater than 1. Got: %.3g", system->gamma);
        return WRAP_RAISE_ERROR(VALUE_ERROR, error_message);
    }

    /* Array pointers */
    if (system->density_)
    {
        return WRAP_RAISE_ERROR(POINTER_ERROR, "Density array pointer (system->density_) is not NULL.");
    }
    if (system->velocity_x_)
    {
        return WRAP_RAISE_ERROR(POINTER_ERROR, "Velocity (x-component) array pointer (system->velocity_x_) is not NULL.");
    }
    if (system->velocity_y_)
    {
        return WRAP_RAISE_ERROR(POINTER_ERROR, "Velocity (y-component) array pointer (system->velocity_y_) is not NULL.");
    }
    if (system->velocity_z_)
    {
        return WRAP_RAISE_ERROR(POINTER_ERROR, "Velocity (z-component) array pointer (system->velocity_z_) is not NULL.");
    }
    if (system->pressure_)
    {
        return WRAP_RAISE_ERROR(POINTER_ERROR, "Pressure array pointer (system->pressure_) is not NULL.");
    }
    if (system->mass_)
    {
        return WRAP_RAISE_ERROR(POINTER_ERROR, "Mass array pointer (system->mass_) is not NULL.");
    }
    if (system->momentum_x_)
    {
        return WRAP_RAISE_ERROR(POINTER_ERROR, "Momentum (x-component) array pointer (system->momentum_x_) is not NULL.");
    }
    if (system->momentum_y_)
    {
        return WRAP_RAISE_ERROR(POINTER_ERROR, "Momentum (y-component) array pointer (system->momentum_y_) is not NULL.");
    }
    if (system->momentum_z_)
    {
        return WRAP_RAISE_ERROR(POINTER_ERROR, "Momentum (z-component) array pointer (system->momentum_z_) is not NULL.");
    }
    if (system->energy_)
    {
        return WRAP_RAISE_ERROR(POINTER_ERROR, "Energy array pointer (system->energy_) is not NULL.");
    }
    if (system->mid_points_x_)
    {
        return WRAP_RAISE_ERROR(POINTER_ERROR, "Mid points x array pointer (system->mid_points_x_) is not NULL.");
    }
    if (system->mid_points_y_)
    {
        return WRAP_RAISE_ERROR(POINTER_ERROR, "Mid points y array pointer (system->mid_points_y_) is not NULL.");
    }
    if (system->mid_points_z_)
    {
        return WRAP_RAISE_ERROR(POINTER_ERROR, "Mid points z array pointer (system->mid_points_z_) is not NULL.");
    }
    if (system->surface_area_x_)
    {
        return WRAP_RAISE_ERROR(POINTER_ERROR, "Surface area x array pointer (system->surface_area_x_) is not NULL.");
    }
    if (system->surface_area_y_)
    {
        return WRAP_RAISE_ERROR(POINTER_ERROR, "Surface area y array pointer (system->surface_area_y_) is not NULL.");
    }
    if (system->surface_area_z_)
    {
        return WRAP_RAISE_ERROR(POINTER_ERROR, "Surface area z array pointer (system->surface_area_z_) is not NULL.");
    }
    if (system->volume_)
    {
        return WRAP_RAISE_ERROR(POINTER_ERROR, "Volume array pointer (system->volume_) is not NULL.");
    }

    /* Number of ghost cells */
    if (system->num_ghost_cells_side < 1)
    {
        size_t error_message_size = (
            strlen("Number of ghost cells per side must be at least 1. Got: ")
            + snprintf(NULL, 0, "%d", system->num_ghost_cells_side)
            + 1 // for the null terminator
        );
        char *error_message = malloc(error_message_size * sizeof(char));
        if (!error_message)
        {
            return WRAP_RAISE_ERROR(MEMORY_ERROR, "Failed to allocate memory for error message.");
        }
        snprintf(error_message, error_message_size, "Number of ghost cells per side must be at least 1. Got: %d", system->num_ghost_cells_side);
        return WRAP_RAISE_ERROR(VALUE_ERROR, error_message);
    }

    /* Number of cells and domain */
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
                size_t error_message_size = strlen("z_min must be less than z_max. Got: z_min = , z_max = ") + 1 + 2 * 128;
                char error_message[error_message_size];
                snprintf(error_message, error_message_size, "z_min must be less than z_max. Got: z_min = %.3g, z_max = %.3g", system->z_min, system->z_max);
                return WRAP_RAISE_ERROR(VALUE_ERROR, error_message);
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
            break;
        default:
            return WRAP_RAISE_ERROR(VALUE_ERROR, "Coordinate system flag not recognized.");
    }

    return make_success_error_status();
}

/**
 * \brief Set the coordinate system flag based on the coordinate system string in the system struct.
 * 
 * \param system Pointer to the system struct.
 * 
 * \return Error status.
 */
IN_FILE ErrorStatus get_coord_sys_flag(System *__restrict system)
{
    const char *coord_sys = system->coord_sys;
    int *coord_sys_flag = &system->coord_sys_flag_;

    if (!coord_sys)
    {
        return WRAP_RAISE_ERROR(POINTER_ERROR, "Coordinate system pointer is NULL.");
    }
    if (!coord_sys_flag)
    {
        return WRAP_RAISE_ERROR(POINTER_ERROR, "Coordinate system flag pointer is NULL.");
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
 * \brief Initialize the cell width for the system struct.
 * 
 * \param system Pointer to the system struct.
 * 
 * \return Error status.
 */
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

/**
 * \brief Initialize the mid points for the system struct.
 * 
 * \param system Pointer to the system struct.
 * 
 * \return Error status.
 */
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

/**
 * \brief Initialize the surface area for the system struct.
 * 
 * \param system Pointer to the system struct.
 * 
 * \return Error status.
 */
IN_FILE ErrorStatus initialize_surface_area(System *__restrict system)
{
    switch (system->coord_sys_flag_)
    {
        case COORD_SYS_CARTESIAN_1D:
        {
            const int total_num_cells_x = system->num_cells_x + 2 * system->num_ghost_cells_side;
            for (int i = 0; i < (total_num_cells_x + 1); i++)
            {
                system->surface_area_x_[i] = 1.0;
            }
            return make_success_error_status();
        }
        case COORD_SYS_CYLINDRICAL_1D: 
        {
            const int total_num_cells_x = system->num_cells_x + 2 * system->num_ghost_cells_side;
            const double dx = system->dx_;
            for (int i = 1; i < (total_num_cells_x + 1); i++)
            {
                const double r = system->x_min + i * dx;
                system->surface_area_x_[i] = 2.0 * M_PI * r;
            }
            return make_success_error_status();
        }
        case COORD_SYS_SPHERICAL_1D:
        {
            const int total_num_cells_x = system->num_cells_x + 2 * system->num_ghost_cells_side;
            const double dx = system->dx_;
            for (int i = 0; i < (total_num_cells_x + 1); i++)
            {
                const double r = system->x_min + i * dx;
                system->surface_area_x_[i] = 4.0 * M_PI * r * r;
            }
            return make_success_error_status();
        }
        case COORD_SYS_CARTESIAN_2D:
        {
            const int total_num_cells_x = system->num_cells_x + 2 * system->num_ghost_cells_side;
            const int total_num_cells_y = system->num_cells_y + 2 * system->num_ghost_cells_side;
            const double area_x = system->dy_;
            const double area_y = system->dx_;
            for (int i = 0; i < total_num_cells_x; i++)
            {
                system->surface_area_x_[i] = area_x;
            }
            for (int j = 0; j < total_num_cells_y; j++)
            {
                system->surface_area_y_[j] = area_y;
            }
            return make_success_error_status();
        }
        case COORD_SYS_CARTESIAN_3D:
        {
            const int total_num_cells_x = system->num_cells_x + 2 * system->num_ghost_cells_side;
            const int total_num_cells_y = system->num_cells_y + 2 * system->num_ghost_cells_side;
            const int total_num_cells_z = system->num_cells_z + 2 * system->num_ghost_cells_side;
            const double area_x = system->dy_ * system->dz_;
            const double area_y = system->dx_ * system->dz_;
            const double area_z = system->dx_ * system->dy_;
            for (int i = 0; i < total_num_cells_x; i++)
            {
                system->surface_area_x_[i] = area_x;
            }
            for (int i = 0; i < total_num_cells_y; i++)
            {
                system->surface_area_y_[i] = area_y;
            }
            for (int i = 0; i < total_num_cells_z; i++)
            {
                system->surface_area_z_[i] = area_z;
            }
            return make_success_error_status();
        }
        default:
        {
            return WRAP_RAISE_ERROR(VALUE_ERROR, "Coordinate system flag not recognized.");
        }
    }
}

/**
 * \brief Initialize the volume for the system struct.
 * 
 * \param system Pointer to the system struct.
 * 
 * \return Error status.
 */
IN_FILE ErrorStatus initialize_volume(System *__restrict system)
{
    switch (system->coord_sys_flag_)
    {
        case COORD_SYS_CARTESIAN_1D:
        {
            const int total_num_cells_x = system->num_cells_x + 2 * system->num_ghost_cells_side;
            const double volume = system->dx_;
            for (int i = 0; i < total_num_cells_x; i++)
            {
                system->volume_[i] = volume;
            }
            return make_success_error_status();
        }
        case COORD_SYS_CYLINDRICAL_1D: 
        {
            const int total_num_cells_x = system->num_cells_x + 2 * system->num_ghost_cells_side;
            const double dx = system->dx_;
            for (int i = 0; i < total_num_cells_x; i++)
            {
                const double r_min = system->x_min + i * dx;
                const double r_max = system->x_min + (i + 1) * dx;
                system->volume_[i] = M_PI * (r_max * r_max - r_min * r_min);
            }
            return make_success_error_status();
        }
        case COORD_SYS_SPHERICAL_1D:
        {
            const int total_num_cells_x = system->num_cells_x + 2 * system->num_ghost_cells_side;
            const double dx = system->dx_;
            for (int i = 0; i < total_num_cells_x; i++)
            {
                const double r_min = system->x_min + i * dx;
                const double r_max = system->x_min + (i + 1) * dx;
                system->volume_[i] = (4.0 / 3.0) * M_PI * (r_max * r_max * r_max - r_min * r_min * r_min);
            }
            return make_success_error_status();
        }
        case COORD_SYS_CARTESIAN_2D:
        {
            const int total_num_cells_x = system->num_cells_x + 2 * system->num_ghost_cells_side;
            const int total_num_cells_y = system->num_cells_y + 2 * system->num_ghost_cells_side;
            const double volume = system->dx_ * system->dy_;
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
            const double volume = system->dx_ * system->dy_ * system->dz_;
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
    system->density_ = calloc(total_num_cells, sizeof(double));
    system->pressure_ = calloc(total_num_cells, sizeof(double));
    system->mass_ = calloc(total_num_cells, sizeof(double));
    system->energy_ = calloc(total_num_cells, sizeof(double));
    system->volume_ = malloc(total_num_cells * sizeof(double));

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
            system->mid_points_z_ = malloc(total_num_cells_z * sizeof(double));
            system->surface_area_z_ = malloc((total_num_cells_z + 1) * sizeof(double));
            system->velocity_z_ = calloc(total_num_cells, sizeof(double));
            system->momentum_z_ = calloc(total_num_cells, sizeof(double));
            if (
                !system->mid_points_z_
                || !system->surface_area_z_
                || !system->velocity_z_
                || !system->momentum_z_
            )
            {
                error_status = WRAP_RAISE_ERROR(MEMORY_ERROR, "Memory allocation failed.");
                goto err_init_memory_alloc_z;
            }
        }
            /* FALL THROUGH */
        case COORD_SYS_CARTESIAN_2D:
        {
            const int total_num_cells_y = system->num_cells_y + 2 * system->num_ghost_cells_side;
            system->mid_points_y_ = malloc(total_num_cells_y * sizeof(double));
            system->surface_area_y_ = malloc((total_num_cells_y + 1) * sizeof(double));
            system->velocity_y_ = calloc(total_num_cells, sizeof(double));
            system->momentum_y_ = calloc(total_num_cells, sizeof(double));
            if (
                !system->mid_points_y_
                || !system->surface_area_y_
                || !system->velocity_y_
                || !system->momentum_y_
            )
            {
                error_status = WRAP_RAISE_ERROR(MEMORY_ERROR, "Memory allocation failed.");
                goto err_init_memory_alloc_y;
            }
        }
            /* FALL THROUGH */
        case COORD_SYS_CARTESIAN_1D: case COORD_SYS_CYLINDRICAL_1D: case COORD_SYS_SPHERICAL_1D:
        {
            const int total_num_cells_x = system->num_cells_x + 2 * system->num_ghost_cells_side;
            system->mid_points_x_ = malloc(total_num_cells_x * sizeof(double));
            system->surface_area_x_ = malloc((total_num_cells_x + 1) * sizeof(double));
            system->velocity_x_ = calloc(total_num_cells, sizeof(double));
            system->momentum_x_ = calloc(total_num_cells, sizeof(double));
            if (
                !system->mid_points_x_
                || !system->surface_area_x_
                || !system->velocity_x_
                || !system->momentum_x_
            )
            {
                error_status = WRAP_RAISE_ERROR(MEMORY_ERROR, "Memory allocation failed.");
                goto err_init_memory_alloc_x;
            }
        }
            break;
        default:
            error_status = WRAP_RAISE_ERROR(VALUE_ERROR, "Coordinate system flag not recognized.");
            goto err_unknown_coord_sys_flag_malloc;
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
    error_status = WRAP_TRACEBACK(initialize_surface_area(system));
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
    free(system->surface_area_z_);
    free(system->velocity_z_);
    free(system->momentum_z_);
err_init_memory_alloc_y:
    free(system->mid_points_y_);
    free(system->surface_area_y_);
    free(system->velocity_y_);
    free(system->momentum_y_);
err_init_memory_alloc_x:
    free(system->mid_points_x_);
    free(system->surface_area_x_);
    free(system->velocity_x_);
    free(system->momentum_x_);
err_unknown_coord_sys_flag_malloc:
err_init_memory_alloc:
    free(system->density_);
    free(system->pressure_);
    free(system->mass_);
    free(system->energy_);
    free(system->volume_);
err_unknown_coord_sys_flag_total_num_cells:
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
    free(system->surface_area_x_);
    free(system->surface_area_y_);
    free(system->surface_area_z_);
    free(system->volume_);
}

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
)
{
#ifdef USE_OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (int i = 0; i < (num_cells_x + 2 * num_ghost_cells_side); i++)
    {
        const double mass_i = mass[i];
        const double momentum_x_i = momentum_x[i];
        const double energy_i = energy[i];
        const double volume_i = volume[i];
        density[i] = mass_i / volume_i;
        velocity_x[i] = momentum_x_i / mass_i;
        pressure[i] = (gamma - 1.0) * (energy_i - 0.5 * mass_i * velocity_x[i] * velocity_x[i]) / volume_i;
    }
}

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
)
{
#ifdef USE_OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (int i = 0; i < (num_cells_x + 2 * num_ghost_cells_side); i++)
    {
        for (int j = 0; j < (num_cells_y + 2 * num_ghost_cells_side); j++)
        {
            const int index = j * (num_cells_x + 2 * num_ghost_cells_side) + i;
            const double mass_ij = mass[index];
            const double momentum_x_ij = momentum_x[index];
            const double momentum_y_ij = momentum_y[index];
            const double energy_ij = energy[index];
            const double volume_ij = volume[index];
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
}

ErrorStatus convert_conserved_to_primitive(System *__restrict system)
{
    switch(system->coord_sys_flag_)
    {
        case COORD_SYS_CARTESIAN_1D: case COORD_SYS_CYLINDRICAL_1D: case COORD_SYS_SPHERICAL_1D:
            convert_conserved_to_primitive_1d(
                system->num_cells_x,
                system->num_ghost_cells_side,
                system->gamma,
                system->volume_,
                system->mass_,
                system->momentum_x_,
                system->energy_,
                system->density_,
                system->velocity_x_,
                system->pressure_
            );
            break;
        case COORD_SYS_CARTESIAN_2D:
            convert_conserved_to_primitive_2d(
                system->num_cells_x,
                system->num_cells_y,
                system->num_ghost_cells_side,
                system->gamma,
                system->volume_,
                system->mass_,
                system->momentum_x_,
                system->momentum_y_,
                system->energy_,
                system->density_,
                system->velocity_x_,
                system->velocity_y_,
                system->pressure_
            );
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
    const double gamma = system->gamma;
    const double *__restrict volume = system->volume_;
    const double *__restrict density = system->density_;
    const double *__restrict velocity_x = system->velocity_x_;
    const double *__restrict velocity_y = system->velocity_y_;
    // const double *__restrict velocity_z = system->velocity_z_;
    const double *__restrict pressure = system->pressure_;
    double *__restrict mass = system->mass_;
    double *__restrict momentum_x = system->momentum_x_;
    double *__restrict momentum_y = system->momentum_y_;
    // double *__restrict momentum_z = system->momentum_z_;
    double *__restrict energy = system->energy_;

    switch(system->coord_sys_flag_)
    {
        case COORD_SYS_CARTESIAN_1D: case COORD_SYS_CYLINDRICAL_1D: case COORD_SYS_SPHERICAL_1D:
#ifdef USE_OPENMP
            #pragma omp parallel for schedule(static)
#endif
            for (int i = 0; i < (system->num_cells_x + 2 * num_ghost_cells_side); i++)
            {
                const double density_i = density[i];
                const double velocity_x_i = velocity_x[i];
                const double pressure_i = pressure[i];
                const double volume_i = volume[i];
                mass[i] = density_i * volume_i;
                momentum_x[i] = mass[i] * velocity_x_i;
                energy[i] = volume[i] * (
                    0.5 * density_i * velocity_x_i * velocity_x_i + pressure_i / (gamma - 1.0)
                );
            }
            break;
        case COORD_SYS_CARTESIAN_2D:
#ifdef USE_OPENMP
            #pragma omp parallel for schedule(static)
#endif
            for (int i = 0; i < (system->num_cells_x + 2 * num_ghost_cells_side); i++)
            {
                for (int j = 0; j < (system->num_cells_y + 2 * num_ghost_cells_side); j++)
                {
                    const int index = j * (system->num_cells_x + 2 * num_ghost_cells_side) + i;
                    const double density_ij = density[index];
                    const double velocity_x_ij = velocity_x[index];
                    const double velocity_y_ij = velocity_y[index];
                    const double pressure_ij = pressure[index];
                    const double volume_ij = volume[index];
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
