/**
 * \file reconstruction.c
 * 
 * \brief Reconstruction functions for the cell interface for the finite volume method.
 * 
 * \author Ching-Yin Ng
 * \date 2025-03-18
 */

#include <string.h>

#include "hydro.h"
#include "reconstruction.h"


ErrorStatus get_reconstruction_flag(IntegratorParam *__restrict integrator_param)
{
    if (!integrator_param->reconstruction)
    {
        return WRAP_RAISE_ERROR(POINTER_ERROR, "Reconstruction is not set.");
    }
    if (strcmp(integrator_param->reconstruction, "piecewise_constant") == 0)
    {
        integrator_param->reconstruction_flag_ = RECONSTRUCTION_PIECEWISE_CONSTANT;
        return make_success_error_status();
    }
    // else if (strcmp(integrator_param->reconstruction, "piecewise_linear") == 0)
    // {
    //     integrator_param->reconstruction_flag_ = RECONSTRUCTION_PIECEWISE_LINEAR;
    //     return make_success_error_status();
    // }
    // else if (strcmp(integrator_param->reconstruction, "piecewise_parabolic") == 0)
    // {
    //     integrator_param->reconstruction_flag_ = RECONSTRUCTION_PIECEWISE_PARABOLIC;
    //     return make_success_error_status();
    // }
    else
    {
        return WRAP_RAISE_ERROR(VALUE_ERROR, "Reconstruction not recognized.");
    }
}

IN_FILE ErrorStatus reconstruct_cell_interface_piecewise_constant_1d(System *__restrict system)
{
    const int total_num_cells_x = system->num_cells_x + 2 * system->num_ghost_cells_side;
    memcpy(system->interface_density_x_, system->density_, total_num_cells_x * sizeof(double));
    memcpy(system->interface_velocity_x_x_, system->velocity_x_, total_num_cells_x * sizeof(double));
    memcpy(system->interface_pressure_x_, system->pressure_, total_num_cells_x * sizeof(double));
    return make_success_error_status();
}

IN_FILE ErrorStatus reconstruct_cell_interface_piecewise_constant_2d(System *__restrict system)
{
    const int total_num_cells_x = system->num_cells_x + 2 * system->num_ghost_cells_side;
    const int total_num_cells_y = system->num_cells_y + 2 * system->num_ghost_cells_side;
    const int total_num_cells = total_num_cells_x * total_num_cells_y;
    memcpy(system->interface_density_x_, system->density_, total_num_cells * sizeof(double));
    memcpy(system->interface_density_y_, system->density_, total_num_cells * sizeof(double));
    memcpy(system->interface_velocity_x_x_, system->velocity_x_, total_num_cells * sizeof(double));
    memcpy(system->interface_velocity_x_y_, system->velocity_y_, total_num_cells * sizeof(double));
    memcpy(system->interface_velocity_y_x_, system->velocity_x_, total_num_cells * sizeof(double));
    memcpy(system->interface_velocity_y_y_, system->velocity_y_, total_num_cells * sizeof(double));
    memcpy(system->interface_pressure_x_, system->pressure_, total_num_cells * sizeof(double));
    memcpy(system->interface_pressure_y_, system->pressure_, total_num_cells * sizeof(double));
    return make_success_error_status();
}

ErrorStatus reconstruct_cell_interface_1d(
    System *__restrict system,
    IntegratorParam *__restrict integrator_param
)
{
    switch (integrator_param->reconstruction_flag_)
    {
        case RECONSTRUCTION_PIECEWISE_CONSTANT:
            return reconstruct_cell_interface_piecewise_constant_1d(system);
        // case RECONSTRUCTION_PIECEWISE_LINEAR:
        //     return reconstruct_cell_interface_piecewise_linear_1d(system, interface_field);
        // case RECONSTRUCTION_PIECEWISE_PARABOLIC:
        //     return reconstruct_cell_interface_piecewise_parabolic_1d(system, interface_field);
        default:
            return WRAP_RAISE_ERROR(VALUE_ERROR, "Reconstruction flag not recognized.");
    }
}

ErrorStatus reconstruct_cell_interface_2d(
    System *__restrict system,
    IntegratorParam *__restrict integrator_param
)
{
    switch (integrator_param->reconstruction_flag_)
    {
        case RECONSTRUCTION_PIECEWISE_CONSTANT:
            return reconstruct_cell_interface_piecewise_constant_2d(system);
        // case RECONSTRUCTION_PIECEWISE_LINEAR:
        //     return reconstruct_cell_interface_piecewise_linear_1d(system, interface_field);
        // case RECONSTRUCTION_PIECEWISE_PARABOLIC:
        //     return reconstruct_cell_interface_piecewise_parabolic_1d(system, interface_field);
        default:
            return WRAP_RAISE_ERROR(VALUE_ERROR, "Reconstruction flag not recognized.");
    }
}

