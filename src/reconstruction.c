/**
 * \file reconstruction.c
 * 
 * \brief Reconstruction functions for the cell interface for the finite volume method.
 * 
 * \author Ching-Yin Ng
 * \date 2025-03-19
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
    const int num_ghost_cells_side = system->num_ghost_cells_side;
    const int num_interfaces = system->num_cells_x + 1;
    memcpy(system->interface_density_x_L_, &(system->density_[num_ghost_cells_side - 1]), num_interfaces * sizeof(double));
    memcpy(system->interface_density_x_R_, &(system->density_[num_ghost_cells_side]), num_interfaces * sizeof(double));
    memcpy(system->interface_velocity_x_x_L_, &(system->velocity_x_[num_ghost_cells_side - 1]), num_interfaces * sizeof(double));
    memcpy(system->interface_velocity_x_x_R_, &(system->velocity_x_[num_ghost_cells_side]), num_interfaces * sizeof(double));
    memcpy(system->interface_pressure_x_L_, &(system->pressure_[num_ghost_cells_side - 1]), num_interfaces * sizeof(double));
    memcpy(system->interface_pressure_x_R_, &(system->pressure_[num_ghost_cells_side]), num_interfaces * sizeof(double));
    return make_success_error_status();
}

IN_FILE ErrorStatus reconstruct_cell_interface_piecewise_constant_2d(System *__restrict system)
{
    const int total_num_cells_x = system->num_cells_x + 2 * system->num_ghost_cells_side;
    const int total_num_cells_y = system->num_cells_y + 2 * system->num_ghost_cells_side;
    const int num_interfaces_x = system->num_cells_x + 1;
    const int num_interfaces_y = system->num_cells_y + 1;
    const int num_ghost_cells_side = system->num_ghost_cells_side;

    double *__restrict density = system->density_;
    double *__restrict velocity_x = system->velocity_x_;
    double *__restrict velocity_y = system->velocity_y_;
    double *__restrict pressure = system->pressure_;

    double *__restrict interface_density_x_L = system->interface_density_x_L_;
    double *__restrict interface_density_x_R = system->interface_density_x_R_;
    double *__restrict interface_density_y_B = system->interface_density_y_B_;
    double *__restrict interface_density_y_T = system->interface_density_y_T_;
    double *__restrict interface_velocity_x_x_L = system->interface_velocity_x_x_L_;
    double *__restrict interface_velocity_x_x_R = system->interface_velocity_x_x_R_;
    double *__restrict interface_velocity_x_y_L = system->interface_velocity_x_y_L_;
    double *__restrict interface_velocity_x_y_R = system->interface_velocity_x_y_R_;
    double *__restrict interface_velocity_y_x_B = system->interface_velocity_y_x_B_;
    double *__restrict interface_velocity_y_x_T = system->interface_velocity_y_x_T_;
    double *__restrict interface_velocity_y_y_B = system->interface_velocity_y_y_B_;
    double *__restrict interface_velocity_y_y_T = system->interface_velocity_y_y_T_;
    double *__restrict interface_pressure_x_L = system->interface_pressure_x_L_;
    double *__restrict interface_pressure_x_R = system->interface_pressure_x_R_;
    double *__restrict interface_pressure_y_B = system->interface_pressure_y_B_;
    double *__restrict interface_pressure_y_T = system->interface_pressure_y_T_;

    for (int i = 0; i < num_interfaces_x; i++)
    {
        for (int j = 0; j < num_interfaces_y; j++)
        {
            const int idx_i = num_ghost_cells_side + i;
            const int idx_j = num_ghost_cells_side + j;
            const int idx_interface = j * num_interfaces_x + i;
            interface_density_x_L[idx_interface] = density[idx_j * total_num_cells_x + (idx_i - 1)];
            interface_density_x_R[idx_interface] = density[idx_j * total_num_cells_x + idx_i];
            interface_velocity_x_x_L[idx_interface] = velocity_x[idx_j * total_num_cells_x + (idx_i - 1)];
            interface_velocity_x_x_R[idx_interface] = velocity_x[idx_j * total_num_cells_x + idx_i];
            interface_velocity_x_y_L[idx_interface] = velocity_y[idx_j * total_num_cells_x + (idx_i - 1)];
            interface_velocity_x_y_R[idx_interface] = velocity_y[idx_j * total_num_cells_x + idx_i];
            interface_pressure_x_L[idx_interface] = pressure[idx_j * total_num_cells_x + (idx_i - 1)];
            interface_pressure_x_R[idx_interface] = pressure[idx_j * total_num_cells_x + idx_i];

            interface_density_y_B[idx_interface] = density[(idx_j - 1) * total_num_cells_x + idx_i];
            interface_density_y_T[idx_interface] = density[idx_j * total_num_cells_x + idx_i];
            interface_velocity_y_x_B[idx_interface] = velocity_x[(idx_j - 1) * total_num_cells_x + idx_i];
            interface_velocity_y_x_T[idx_interface] = velocity_x[idx_j * total_num_cells_x + idx_i];
            interface_velocity_y_y_B[idx_interface] = velocity_y[(idx_j - 1) * total_num_cells_x + idx_i];
            interface_velocity_y_y_T[idx_interface] = velocity_y[idx_j * total_num_cells_x + idx_i];
            interface_pressure_y_B[idx_interface] = pressure[(idx_j - 1) * total_num_cells_x + idx_i];
            interface_pressure_y_T[idx_interface] = pressure[idx_j * total_num_cells_x + idx_i];
        }
    }

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
        //     return reconstruct_cell_interface_piecewise_linear_1d(system);
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

