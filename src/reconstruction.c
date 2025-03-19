/**
 * \file reconstruction.c
 * 
 * \brief Reconstruction functions for the cell interface for the finite volume method.
 * 
 * \author Ching-Yin Ng
 * \date 2025-03-19
 */

#include <string.h>

#ifdef USE_OPENMP
#include <omp.h>
#endif

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
    else if (strcmp(integrator_param->reconstruction, "piecewise_linear") == 0)
    {
        integrator_param->reconstruction_flag_ = RECONSTRUCTION_PIECEWISE_LINEAR;
        return make_success_error_status();
    }
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

IN_FILE double van_leer(double a, double b)
{
    if (a * b <= 0.0)
    {
        return 0.0;
    }
    else
    {
        return 2.0 * a * b / (a + b + 1e-10);
    }
}

IN_FILE void reconstruct_cell_interface_piecewise_constant(
    const double *__restrict field,
    double *__restrict interface_field_L,
    double *__restrict interface_field_R,
    const int num_cells,
    const int num_ghost_cells_side
)
{
    const int num_interfaces = num_cells + 1;
    memcpy(interface_field_L, &(field[num_ghost_cells_side - 1]), num_interfaces * sizeof(double));
    memcpy(interface_field_R, &(field[num_ghost_cells_side]), num_interfaces * sizeof(double));
}

IN_FILE void reconstruct_cell_interface_piecewise_linear(
    const double *__restrict field,
    double *__restrict interface_field_L,
    double *__restrict interface_field_R,
    const int num_cells,
    const int num_ghost_cells_side
)
{
    const int num_cells_plus_two = num_cells + 2;

    for (int i = 0; i < num_cells_plus_two; i++)
    {
        const int idx_i = num_ghost_cells_side + i - 1;
        
        // Calculate left and right differences
        const double delta_field_left = field[idx_i] - field[idx_i - 1];
        const double delta_field_right = field[idx_i + 1] - field[idx_i];

        // Apply slope limiter
        const double delta_field = van_leer(delta_field_left, delta_field_right);

        if (i > 0)
        {
            interface_field_R[i - 1] = field[idx_i] - 0.5 * delta_field;
        }
        if (i < num_cells + 1)
        {
            interface_field_L[i] = field[idx_i] + 0.5 * delta_field;
        }
    }
}

void reconstruct_cell_interface(
    const IntegratorParam *__restrict integrator_param,
    const double *__restrict field,
    double *__restrict interface_field_L,
    double *__restrict interface_field_R,
    const int num_cells,
    const int num_ghost_cells_side
)
{
    switch (integrator_param->reconstruction_flag_)
    {
        case RECONSTRUCTION_PIECEWISE_CONSTANT:
            reconstruct_cell_interface_piecewise_constant(
                field,
                interface_field_L,
                interface_field_R,
                num_cells,
                num_ghost_cells_side
            );
            break;
        case RECONSTRUCTION_PIECEWISE_LINEAR:
            reconstruct_cell_interface_piecewise_linear(
                field,
                interface_field_L,
                interface_field_R,
                num_cells,
                num_ghost_cells_side
            );
            break;
        // case RECONSTRUCTION_PIECEWISE_PARABOLIC:
        //     return reconstruct_cell_interface_piecewise_parabolic_1d(system, interface_field);
    }
}
