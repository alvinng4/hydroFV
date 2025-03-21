/**
 * \file boundary.c
 * 
 * \brief Boundary condition functions for the hydrodynamics simulation
 * 
 * \author Ching-Yin Ng
 * \date 2025-03-21
 */

#include <stdio.h>
#include <string.h>

#include "hydro.h"


BoundaryConditionParam get_new_boundary_condition_param(void)
{
    BoundaryConditionParam boundary_condition_param = {
        .boundary_condition_x_min = NULL,
        .boundary_condition_x_max = NULL,
        .boundary_condition_y_min = NULL,
        .boundary_condition_y_max = NULL,
        .boundary_condition_z_min = NULL,
        .boundary_condition_z_max = NULL,
        .set_boundary_condition_1d_x_min = NULL,
        .set_boundary_condition_1d_x_max = NULL,
        .set_boundary_condition_cartesian_2d_x_min = NULL,
        .set_boundary_condition_cartesian_2d_x_max = NULL,
        .set_boundary_condition_cartesian_2d_y_min = NULL,
        .set_boundary_condition_cartesian_2d_y_max = NULL,

        .boundary_condition_flag_x_min_ = -1,
        .boundary_condition_flag_x_max_ = -1,
        .boundary_condition_flag_y_min_ = -1,
        .boundary_condition_flag_y_max_ = -1,
        .boundary_condition_flag_z_min_ = -1,
        .boundary_condition_flag_z_max_ = -1
    };

    return boundary_condition_param;
}

/**
 * \brief Set boundary condition flags based on the boundary condition strings in the system struct.
 * 
 * \param system Pointer to the system struct.
 * \param boundary_condition_param Pointer to the boundary condition parameter struct.
 * 
 * \return Error status.
 */
IN_FILE ErrorStatus get_boundary_condition_flag(
    const System *__restrict system,
    BoundaryConditionParam *__restrict boundary_condition_param
)
{
    ErrorStatus error_status;

    const char* boundary_condition_x_min = boundary_condition_param->boundary_condition_x_min;
    const char* boundary_condition_x_max = boundary_condition_param->boundary_condition_x_max;
    const char* boundary_condition_y_min = boundary_condition_param->boundary_condition_y_min;
    const char* boundary_condition_y_max = boundary_condition_param->boundary_condition_y_max;
    const char* boundary_condition_z_min = boundary_condition_param->boundary_condition_z_min;
    const char* boundary_condition_z_max = boundary_condition_param->boundary_condition_z_max;

    switch (system->coord_sys_flag_)
    {
        /* z-direction */
        case COORD_SYS_CARTESIAN_3D:
            if (!boundary_condition_z_min || !boundary_condition_z_max)
            {
                error_status = WRAP_RAISE_ERROR(POINTER_ERROR, "Boundary condition for z_min or z_max is NULL.");
                goto err_null_boundary_condition;
            }

            if (strcmp(boundary_condition_z_min, "none") == 0)
            {
                boundary_condition_param->boundary_condition_flag_z_min_ = BOUNDARY_CONDITION_NONE;
            }
            else if (strcmp(boundary_condition_z_min, "reflective") == 0)
            {
                boundary_condition_param->boundary_condition_flag_z_min_ = BOUNDARY_CONDITION_REFLECTIVE;
            }
            else if (strcmp(boundary_condition_z_min, "transmissive") == 0)
            {
                boundary_condition_param->boundary_condition_flag_z_min_ = BOUNDARY_CONDITION_TRANSMISSIVE;
            }
            else if (strcmp(boundary_condition_z_min, "periodic") == 0)
            {
                boundary_condition_param->boundary_condition_flag_z_min_ = BOUNDARY_CONDITION_PERIODIC;
            }
            else if (strcmp(boundary_condition_z_min, "custom") == 0)
            {
                boundary_condition_param->boundary_condition_flag_z_min_ = BOUNDARY_CONDITION_CUSTOM;
            }
            else
            {
                error_status = WRAP_RAISE_ERROR(VALUE_ERROR, "Boundary condition for z_min is not recognized.");
                goto err_unknown_boundary_condition;
            }

            if (strcmp(boundary_condition_z_max, "none") == 0)
            {
                boundary_condition_param->boundary_condition_flag_z_max_ = BOUNDARY_CONDITION_NONE;
            }
            else if (strcmp(boundary_condition_z_max, "reflective") == 0)
            {
                boundary_condition_param->boundary_condition_flag_z_max_ = BOUNDARY_CONDITION_REFLECTIVE;
            }
            else if (strcmp(boundary_condition_z_max, "transmissive") == 0)
            {
                boundary_condition_param->boundary_condition_flag_z_max_ = BOUNDARY_CONDITION_TRANSMISSIVE;
            }
            else if (strcmp(boundary_condition_z_max, "periodic") == 0)
            {
                boundary_condition_param->boundary_condition_flag_z_max_ = BOUNDARY_CONDITION_PERIODIC;
            }
            else if (strcmp(boundary_condition_z_max, "custom") == 0)
            {
                boundary_condition_param->boundary_condition_flag_z_max_ = BOUNDARY_CONDITION_CUSTOM;
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
                error_status = WRAP_RAISE_ERROR(POINTER_ERROR, "Boundary condition for y_min or y_max is NULL.");
                goto err_null_boundary_condition;
            }

            if (strcmp(boundary_condition_y_min, "none") == 0)
            {
                boundary_condition_param->boundary_condition_flag_y_min_ = BOUNDARY_CONDITION_NONE;
            }
            else if (strcmp(boundary_condition_y_min, "reflective") == 0)
            {
                boundary_condition_param->boundary_condition_flag_y_min_ = BOUNDARY_CONDITION_REFLECTIVE;
            }
            else if (strcmp(boundary_condition_y_min, "transmissive") == 0)
            {
                boundary_condition_param->boundary_condition_flag_y_min_ = BOUNDARY_CONDITION_TRANSMISSIVE;
            }
            else if (strcmp(boundary_condition_y_min, "periodic") == 0)
            {
                boundary_condition_param->boundary_condition_flag_y_min_ = BOUNDARY_CONDITION_PERIODIC;
            }
            else if (strcmp(boundary_condition_y_min, "custom") == 0)
            {
                boundary_condition_param->boundary_condition_flag_y_min_ = BOUNDARY_CONDITION_CUSTOM;
            }
            else
            {
                error_status = WRAP_RAISE_ERROR(VALUE_ERROR, "Boundary condition for y_min is not recognized.");
                goto err_unknown_boundary_condition;
            }

            if (strcmp(boundary_condition_y_max, "none") == 0)
            {
                boundary_condition_param->boundary_condition_flag_y_max_ = BOUNDARY_CONDITION_NONE;
            }
            else if (strcmp(boundary_condition_y_max, "reflective") == 0)
            {
                boundary_condition_param->boundary_condition_flag_y_max_ = BOUNDARY_CONDITION_REFLECTIVE;
            }
            else if (strcmp(boundary_condition_y_max, "transmissive") == 0)
            {
                boundary_condition_param->boundary_condition_flag_y_max_ = BOUNDARY_CONDITION_TRANSMISSIVE;
            }
            else if (strcmp(boundary_condition_y_max, "periodic") == 0)
            {
                boundary_condition_param->boundary_condition_flag_y_max_ = BOUNDARY_CONDITION_PERIODIC;
            }
            else if (strcmp(boundary_condition_y_max, "custom") == 0)
            {
                boundary_condition_param->boundary_condition_flag_y_max_ = BOUNDARY_CONDITION_CUSTOM;
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
                error_status = WRAP_RAISE_ERROR(POINTER_ERROR, "Boundary condition for x_min or x_max is NULL.");
                goto err_null_boundary_condition;
            }

            if (strcmp(boundary_condition_x_min, "none") == 0)
            {
                boundary_condition_param->boundary_condition_flag_x_min_ = BOUNDARY_CONDITION_NONE;
            }
            else if (strcmp(boundary_condition_x_min, "reflective") == 0)
            {
                boundary_condition_param->boundary_condition_flag_x_min_ = BOUNDARY_CONDITION_REFLECTIVE;
            }
            else if (strcmp(boundary_condition_x_min, "transmissive") == 0)
            {
                boundary_condition_param->boundary_condition_flag_x_min_ = BOUNDARY_CONDITION_TRANSMISSIVE;
            }
            else if (strcmp(boundary_condition_x_min, "periodic") == 0)
            {
                boundary_condition_param->boundary_condition_flag_x_min_ = BOUNDARY_CONDITION_PERIODIC;
            }
            else if (strcmp(boundary_condition_x_min, "custom") == 0)
            {
                boundary_condition_param->boundary_condition_flag_x_min_ = BOUNDARY_CONDITION_CUSTOM;
            }
            else
            {
                error_status = WRAP_RAISE_ERROR(VALUE_ERROR, "Boundary condition for x_min is not recognized.");
                goto err_unknown_boundary_condition;
            }

            if (strcmp(boundary_condition_x_max, "none") == 0)
            {
                boundary_condition_param->boundary_condition_flag_x_max_ = BOUNDARY_CONDITION_NONE;
            }
            else if (strcmp(boundary_condition_x_max, "reflective") == 0)
            {
                boundary_condition_param->boundary_condition_flag_x_max_ = BOUNDARY_CONDITION_REFLECTIVE;
            }
            else if (strcmp(boundary_condition_x_max, "transmissive") == 0)
            {
                boundary_condition_param->boundary_condition_flag_x_max_ = BOUNDARY_CONDITION_TRANSMISSIVE;
            }
            else if (strcmp(boundary_condition_x_max, "periodic") == 0)
            {
                boundary_condition_param->boundary_condition_flag_x_max_ = BOUNDARY_CONDITION_PERIODIC;
            }
            else if (strcmp(boundary_condition_x_max, "custom") == 0)
            {
                boundary_condition_param->boundary_condition_flag_x_max_ = BOUNDARY_CONDITION_CUSTOM;
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
                boundary_condition_param->boundary_condition_flag_z_min_ == BOUNDARY_CONDITION_PERIODIC
                || boundary_condition_param->boundary_condition_flag_z_max_ == BOUNDARY_CONDITION_PERIODIC
            )
            {
                if (
                    boundary_condition_param->boundary_condition_flag_z_min_ != boundary_condition_param->boundary_condition_flag_z_max_
                )
                {
                    error_status = WRAP_RAISE_ERROR(VALUE_ERROR, "Periodic boundary condition mismatch for z_min and z_max.");
                    goto err_periodic_boundary_condition_mismatch;
                }
            }
            /* FALL THROUGH */

        case COORD_SYS_CARTESIAN_2D:
            if (
                boundary_condition_param->boundary_condition_flag_y_min_ == BOUNDARY_CONDITION_PERIODIC
                || boundary_condition_param->boundary_condition_flag_y_max_ == BOUNDARY_CONDITION_PERIODIC
            )
            {
                if (
                    boundary_condition_param->boundary_condition_flag_y_min_ != boundary_condition_param->boundary_condition_flag_y_max_
                )
                {
                    error_status = WRAP_RAISE_ERROR(VALUE_ERROR, "Periodic boundary condition mismatch for y_min and y_max.");
                    goto err_periodic_boundary_condition_mismatch;
                }
            }
            /* FALL THROUGH */

        case COORD_SYS_CARTESIAN_1D: case COORD_SYS_CYLINDRICAL_1D: case COORD_SYS_SPHERICAL_1D:
            if (
                boundary_condition_param->boundary_condition_flag_x_min_ == BOUNDARY_CONDITION_PERIODIC
                || boundary_condition_param->boundary_condition_flag_x_max_ == BOUNDARY_CONDITION_PERIODIC
            )
            {
                if (
                    boundary_condition_param->boundary_condition_flag_x_min_ != boundary_condition_param->boundary_condition_flag_x_max_
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

    /* Check for custom boundary condition */
    switch (system->coord_sys_flag_)
    {
        case COORD_SYS_CARTESIAN_1D: case COORD_SYS_CYLINDRICAL_1D: case COORD_SYS_SPHERICAL_1D:
            if (
                boundary_condition_param->boundary_condition_flag_x_min_ == BOUNDARY_CONDITION_CUSTOM
                && !boundary_condition_param->set_boundary_condition_1d_x_min
            )
            {
                error_status = WRAP_RAISE_ERROR(POINTER_ERROR, "Custom boundary condition pointer for x_min is NULL.");
                goto err_null_custom_boundary_condition_pointer;
            }

            if (
                boundary_condition_param->boundary_condition_flag_x_max_ == BOUNDARY_CONDITION_CUSTOM
                && !boundary_condition_param->set_boundary_condition_1d_x_max
            )
            {
                error_status = WRAP_RAISE_ERROR(POINTER_ERROR, "Custom boundary condition pointer for x_max is NULL.");
                goto err_null_custom_boundary_condition_pointer;
            }
            break;
        case COORD_SYS_CARTESIAN_2D:
            if (
                boundary_condition_param->boundary_condition_flag_x_min_ == BOUNDARY_CONDITION_CUSTOM
                && !boundary_condition_param->set_boundary_condition_cartesian_2d_x_min
            )
            {
                error_status = WRAP_RAISE_ERROR(POINTER_ERROR, "Custom boundary condition pointer for x_min is NULL.");
                goto err_null_custom_boundary_condition_pointer;
            }

            if (
                boundary_condition_param->boundary_condition_flag_x_max_ == BOUNDARY_CONDITION_CUSTOM
                && !boundary_condition_param->set_boundary_condition_cartesian_2d_x_max
            )
            {
                error_status = WRAP_RAISE_ERROR(POINTER_ERROR, "Custom boundary condition pointer for x_max is NULL.");
                goto err_null_custom_boundary_condition_pointer;
            }

            if (
                boundary_condition_param->boundary_condition_flag_y_min_ == BOUNDARY_CONDITION_CUSTOM
                && !boundary_condition_param->set_boundary_condition_cartesian_2d_y_min
            )
            {
                error_status = WRAP_RAISE_ERROR(POINTER_ERROR, "Custom boundary condition pointer for y_min is NULL.");
                goto err_null_custom_boundary_condition_pointer;
            }

            if (
                boundary_condition_param->boundary_condition_flag_y_max_ == BOUNDARY_CONDITION_CUSTOM
                && !boundary_condition_param->set_boundary_condition_cartesian_2d_y_max
            )
            {
                error_status = WRAP_RAISE_ERROR(POINTER_ERROR, "Custom boundary condition pointer for y_max is NULL.");
                goto err_null_custom_boundary_condition_pointer;
            }
            break;
        case COORD_SYS_CARTESIAN_3D:
            error_status = WRAP_RAISE_ERROR(NOT_IMPLEMENTED_ERROR, "Custom boundary condition for 3D is not implemented.");
            break;
        default:
            error_status = WRAP_RAISE_ERROR(VALUE_ERROR, "Coordinate system flag not recognized.");
            goto err_unknown_coord_sys_flag;
    }

    return make_success_error_status();

err_null_custom_boundary_condition_pointer:
err_periodic_boundary_condition_mismatch:
err_unknown_coord_sys_flag:
err_unknown_boundary_condition:
err_null_boundary_condition:
    return error_status;
}

ErrorStatus finalize_boundary_condition_param(
    const System *__restrict system,
    BoundaryConditionParam *__restrict boundary_condition_param
)
{
    ErrorStatus error_status;

    /* Get boundary condition flags */
    error_status = WRAP_TRACEBACK(get_boundary_condition_flag(system, boundary_condition_param));
    if (error_status.return_code != SUCCESS)
    {
        goto err_boundary_condition_flag;
    }

    return error_status;

err_boundary_condition_flag:
    return error_status;
}

ErrorStatus set_boundary_condition_1d(
    const BoundaryConditionParam *__restrict boundary_condition_param,
    double *__restrict density,
    double *__restrict velocity_x,
    double *__restrict pressure,
    const int num_ghost_cells_side,
    const int num_cells_x
)
{
    ErrorStatus error_status;

    switch (boundary_condition_param->boundary_condition_flag_x_min_)
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
        case BOUNDARY_CONDITION_CUSTOM:
        {
            boundary_condition_param->set_boundary_condition_1d_x_min(
                density,
                velocity_x,
                pressure,
                num_ghost_cells_side,
                num_cells_x
            );
        }
            break;
        default:
            error_status = WRAP_RAISE_ERROR(VALUE_ERROR, "Boundary condition flag not recognized for x_min.");
            goto err_unknown_boundary_condition_flag;
    }

    switch (boundary_condition_param->boundary_condition_flag_x_max_)
    {
        case BOUNDARY_CONDITION_NONE:
            break;
        case BOUNDARY_CONDITION_REFLECTIVE:
            for (int i = 0; i < num_ghost_cells_side; i++)
            {
                density[num_ghost_cells_side + num_cells_x + i] = density[num_ghost_cells_side + num_cells_x - 1 - i];
                velocity_x[num_ghost_cells_side + num_cells_x + i] = -velocity_x[num_ghost_cells_side + num_cells_x - 1 - i];
                pressure[num_ghost_cells_side + num_cells_x + i] = pressure[num_ghost_cells_side + num_cells_x - 1 - i];
            }
            break;
        case BOUNDARY_CONDITION_TRANSMISSIVE:
            for (int i = 0; i < num_ghost_cells_side; i++)
            {
                density[num_ghost_cells_side + num_cells_x + i] = density[num_ghost_cells_side + num_cells_x - 1 - i];
                velocity_x[num_ghost_cells_side + num_cells_x + i] = velocity_x[num_ghost_cells_side + num_cells_x - 1 - i];
                pressure[num_ghost_cells_side + num_cells_x + i] = pressure[num_ghost_cells_side + num_cells_x - 1 - i];
            }
            break;
        case BOUNDARY_CONDITION_PERIODIC:
            for (int i = 0; i < num_ghost_cells_side; i++)
            {
                density[num_ghost_cells_side + num_cells_x + i] = density[num_ghost_cells_side + i];
                velocity_x[num_ghost_cells_side + num_cells_x + i] = velocity_x[num_ghost_cells_side + i];
                pressure[num_ghost_cells_side + num_cells_x + i] = pressure[num_ghost_cells_side + i];

                density[num_ghost_cells_side - 1 - i] = density[num_ghost_cells_side + num_cells_x - 1 - i];
                velocity_x[num_ghost_cells_side - 1 - i] = velocity_x[num_ghost_cells_side + num_cells_x - 1 - i];
                pressure[num_ghost_cells_side - 1 - i] = pressure[num_ghost_cells_side + num_cells_x - 1 - i];
            }
            break;
        case BOUNDARY_CONDITION_CUSTOM:
            boundary_condition_param->set_boundary_condition_1d_x_max(
                density,
                velocity_x,
                pressure,
                num_ghost_cells_side,
                num_cells_x
            );
            break;
        default:
            error_status = WRAP_RAISE_ERROR(VALUE_ERROR, "Boundary condition flag not recognized for x_max.");
            goto err_unknown_boundary_condition_flag;
    }

    return make_success_error_status();

err_unknown_boundary_condition_flag:
    return error_status;
}

ErrorStatus set_boundary_condition_cartesian_2d(
    const BoundaryConditionParam *__restrict boundary_condition_param,
    double *__restrict density,
    double *__restrict velocity_x,
    double *__restrict velocity_y,
    double *__restrict pressure,
    const int num_ghost_cells_side,
    const int num_cells_x,
    const int num_cells_y
)
{
    ErrorStatus error_status;

    const int total_num_cells_x = num_cells_x + 2 * num_ghost_cells_side;
    switch (boundary_condition_param->boundary_condition_flag_x_min_)
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
        case BOUNDARY_CONDITION_CUSTOM:
            boundary_condition_param->set_boundary_condition_cartesian_2d_x_min(
                density,
                velocity_x,
                velocity_y,
                pressure,
                num_ghost_cells_side,
                num_cells_x,
                num_cells_y
            );
            break;
        default:
            error_status = WRAP_RAISE_ERROR(VALUE_ERROR, "Boundary condition flag not recognized for x_min.");
            goto err_unknown_boundary_condition_flag;
    }

    switch (boundary_condition_param->boundary_condition_flag_x_max_)
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
        case BOUNDARY_CONDITION_CUSTOM:
            boundary_condition_param->set_boundary_condition_cartesian_2d_x_max(
                density,
                velocity_x,
                velocity_y,
                pressure,
                num_ghost_cells_side,
                num_cells_x,
                num_cells_y
            );
            break;
        default:
            error_status = WRAP_RAISE_ERROR(VALUE_ERROR, "Boundary condition flag not recognized for x_max.");
            goto err_unknown_boundary_condition_flag;
    }

    switch (boundary_condition_param->boundary_condition_flag_y_min_)
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
        case BOUNDARY_CONDITION_CUSTOM:
            boundary_condition_param->set_boundary_condition_cartesian_2d_y_min(
                density,
                velocity_x,
                velocity_y,
                pressure,
                num_ghost_cells_side,
                num_cells_x,
                num_cells_y
            );
            break;
        default:
            error_status = WRAP_RAISE_ERROR(VALUE_ERROR, "Boundary condition flag not recognized for y_min.");
            goto err_unknown_boundary_condition_flag;
    }

    switch (boundary_condition_param->boundary_condition_flag_y_max_)
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
        case BOUNDARY_CONDITION_CUSTOM:
            boundary_condition_param->set_boundary_condition_cartesian_2d_y_max(
                density,
                velocity_x,
                velocity_y,
                pressure,
                num_ghost_cells_side,
                num_cells_x,
                num_cells_y
            );
            break;
        default:
            error_status = WRAP_RAISE_ERROR(VALUE_ERROR, "Boundary condition flag not recognized for y_max.");
            goto err_unknown_boundary_condition_flag;
    }

    return make_success_error_status();

err_unknown_boundary_condition_flag:
    return error_status;
}

/**
 * \brief Set the boundary condition for 3D Cartesian system.
 * 
 * \param system Pointer to the system struct.
 * 
 * \return Error status.
 */
// IN_FILE ErrorStatus set_boundary_condition_cartesian_3d(System *__restrict system)
// {
//     (void) system;
//     return WRAP_RAISE_ERROR(NOT_IMPLEMENTED_ERROR, "");
// }

ErrorStatus set_boundary_condition(
    const BoundaryConditionParam *__restrict boundary_condition_param,
    System *__restrict system
)
{
    switch (system->coord_sys_flag_)
    {
        case COORD_SYS_CARTESIAN_1D: case COORD_SYS_CYLINDRICAL_1D: case COORD_SYS_SPHERICAL_1D:
            return WRAP_TRACEBACK(set_boundary_condition_1d(
                boundary_condition_param,
                system->density_,
                system->velocity_x_,
                system->pressure_,
                system->num_ghost_cells_side,
                system->num_cells_x
            ));
        case COORD_SYS_CARTESIAN_2D:
            return WRAP_TRACEBACK(set_boundary_condition_cartesian_2d(
                boundary_condition_param,
                system->density_,
                system->velocity_x_,
                system->velocity_y_,
                system->pressure_,
                system->num_ghost_cells_side,
                system->num_cells_x,
                system->num_cells_y
            ));
        // case COORD_SYS_CARTESIAN_3D:
        //     return WRAP_TRACEBACK(set_boundary_condition_cartesian_3d(system));
        default:
            return WRAP_RAISE_ERROR(VALUE_ERROR, "Coordinate system flag not recognized.");
    }
}
