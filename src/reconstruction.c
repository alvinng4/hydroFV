/**
 * \file reconstruction.c
 * 
 * \brief Reconstruction functions for the cell interface for the finite volume method.
 * 
 * \author Ching-Yin Ng
 * \date 2025-03-20
 */

#include <math.h>
#include <string.h>

#ifdef USE_OPENMP
#include <omp.h>
#endif

#include "hydro.h"
#include "reconstruction.h"

typedef double (*reconstruction_limiter_func)(const double, const double);


ErrorStatus get_reconstruction_flag(IntegratorParam *__restrict integrator_param)
{
    if (!integrator_param->reconstruction)
    {
        return WRAP_RAISE_ERROR(POINTER_ERROR, "Reconstruction is not set.");
    }
    if (strcmp(integrator_param->reconstruction, "piecewise_constant") == 0)
    {
        integrator_param->reconstruction_flag_ = RECONSTRUCTION_PIECEWISE_CONSTANT;
    }
    else if (strcmp(integrator_param->reconstruction, "piecewise_linear") == 0)
    {
        integrator_param->reconstruction_flag_ = RECONSTRUCTION_PIECEWISE_LINEAR;
    }
    else if (strcmp(integrator_param->reconstruction, "piecewise_parabolic") == 0)
    {
        integrator_param->reconstruction_flag_ = RECONSTRUCTION_PIECEWISE_PARABOLIC;
    }
    else
    {
        return WRAP_RAISE_ERROR(VALUE_ERROR, "Reconstruction not recognized.");
    }

    if (!integrator_param->reconstruction_limiter)
    {
        return WRAP_RAISE_ERROR(POINTER_ERROR, "Reconstruction limiter is not set.");
    }

    if (strcmp(integrator_param->reconstruction_limiter, "minmod") == 0)
    {
        integrator_param->reconstruction_limiter_flag_ = RECONSTRUCTION_LIMITER_MINMOD;
        return make_success_error_status();
    }
    else if (strcmp(integrator_param->reconstruction_limiter, "van_leer") == 0)
    {
        integrator_param->reconstruction_limiter_flag_ = RECONSTRUCTION_LIMITER_VAN_LEER;
        return make_success_error_status();
    }
    else if (strcmp(integrator_param->reconstruction_limiter, "monotonized_center") == 0)
    {
        integrator_param->reconstruction_limiter_flag_ = RECONSTRUCTION_LIMITER_MONOTONIZED_CENTER;
        return make_success_error_status();
    }
    else
    {
        return WRAP_RAISE_ERROR(VALUE_ERROR, "Reconstruction limiter not recognized.");
    }
}

IN_FILE double minmod(const double a, const double b)
{
    if (a * b <= 0.0)
    {
        return 0.0;
    }
    else if (fabs(a) < fabs(b))
    {
        return a;
    }
    else
    {
        return b;
    }
}

IN_FILE double van_leer(const double a, const double b)
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

IN_FILE double monotonized_center(const double a, const double b)
{
    if (a * b <= 0.0)
    {
        return 0.0;
    }
    else if (a + b > 0)
    {
        return fmin(0.5 * (a + b), 2.0 * fmin(fabs(a), fabs(b)));
    }
    else
    {
        return -fmin(-0.5 * (a + b), 2.0 * fmin(fabs(a), fabs(b)));
    }
}

IN_FILE void reconstruct_cell_interface_piecewise_constant(
    const double *__restrict field,
    double *__restrict interface_field_L,
    double *__restrict interface_field_R,
    const int num_cells,
    const int num_ghost_cells_side,
    reconstruction_limiter_func limiter
)
{
    (void) limiter;

    const int num_interfaces = num_cells + 1;
    memcpy(interface_field_L, &(field[num_ghost_cells_side - 1]), num_interfaces * sizeof(double));
    memcpy(interface_field_R, &(field[num_ghost_cells_side]), num_interfaces * sizeof(double));
}

IN_FILE void reconstruct_cell_interface_piecewise_linear(
    const double *__restrict field,
    double *__restrict interface_field_L,
    double *__restrict interface_field_R,
    const int num_cells,
    const int num_ghost_cells_side,
    reconstruction_limiter_func limiter
)
{
    const int num_cells_plus_two = num_cells + 2;

    for (int i = 0; i < num_cells_plus_two; i++)
    {
        const int idx_i = num_ghost_cells_side + i - 1;

        // Calculate the field difference
        const double delta_field_L = field[idx_i] - field[idx_i - 1];
        const double delta_field_R = field[idx_i + 1] - field[idx_i];

        // Apply slope limiter
        const double delta_field = limiter(delta_field_L, delta_field_R);

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

IN_FILE void reconstruct_cell_interface_piecewise_parabolic(
    const double *__restrict field,
    double *__restrict interface_field_L,
    double *__restrict interface_field_R,
    const int num_cells,
    const int num_ghost_cells_side,
    reconstruction_limiter_func limiter
)
{
    const int num_cells_plus_two = num_cells + 2;

    for (int i = 0; i < num_cells_plus_two; i++)
    {
        const int idx_i = num_ghost_cells_side + i - 1;

        // Calculate the field difference
        const double delta_field_a = field[idx_i - 1] - field[idx_i - 2];
        const double delta_field_b = field[idx_i] - field[idx_i - 1];
        const double delta_field_c = field[idx_i + 1] - field[idx_i];
        const double delta_field_d = field[idx_i + 2] - field[idx_i + 1];

        const double delta_field_i_minus_one = limiter(delta_field_a, delta_field_b);
        const double delta_field_i = limiter(delta_field_b, delta_field_c);
        const double delta_field_i_plus_one = limiter(delta_field_c, delta_field_d);

        double interface_L = (
            0.5 * (field[idx_i - 1] + field[idx_i])
            + (delta_field_i_minus_one - delta_field_i) / 6.0
        );
        double interface_R = (
            0.5 * (field[idx_i] + field[idx_i + 1])
            + (delta_field_i - delta_field_i_plus_one) / 6.0
        );

        // Apply slope limiter
        const double delta_R = interface_R - field[idx_i];
        const double delta_L = field[idx_i] - interface_L;
        const double delta_interface = interface_R - interface_L;
        const double sum_interface = interface_R + interface_L;

        if (delta_R * delta_L <= 0.0)
        {
            interface_L = field[idx_i];
            interface_R = field[idx_i];
        }
        else if (
            delta_interface * (field[idx_i] - 0.5 * sum_interface)
            > (delta_interface * delta_interface) / 6.0
        )
        {
            interface_L = 3.0 * field[idx_i] - 2.0 * interface_R;
        }
        else if (
            -(delta_interface * delta_interface) / 6.0 >
            delta_interface * (field[idx_i] - 0.5 * sum_interface)
        )
        {
            interface_R = 3.0 * field[idx_i] - 2.0 * interface_L;
        }

        if (i > 0)
        {
            interface_field_R[i - 1] = interface_L;
        }
        if (i < num_cells + 1)
        {
            interface_field_L[i] = interface_R;
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
    reconstruction_limiter_func limiter = NULL;
    switch (integrator_param->reconstruction_limiter_flag_)
    {
        case RECONSTRUCTION_LIMITER_MINMOD:
            limiter = minmod;
            break;
        case RECONSTRUCTION_LIMITER_VAN_LEER:
            limiter = van_leer;
            break;
        case RECONSTRUCTION_LIMITER_MONOTONIZED_CENTER:
            limiter = monotonized_center;
            break;
    }

    switch (integrator_param->reconstruction_flag_)
    {
        case RECONSTRUCTION_PIECEWISE_CONSTANT:
            reconstruct_cell_interface_piecewise_constant(
                field,
                interface_field_L,
                interface_field_R,
                num_cells,
                num_ghost_cells_side,
                limiter
            );
            break;
        case RECONSTRUCTION_PIECEWISE_LINEAR:
            reconstruct_cell_interface_piecewise_linear(
                field,
                interface_field_L,
                interface_field_R,
                num_cells,
                num_ghost_cells_side,
                limiter
            );
            break;
        case RECONSTRUCTION_PIECEWISE_PARABOLIC:
            reconstruct_cell_interface_piecewise_parabolic(
                field,
                interface_field_L,
                interface_field_R,
                num_cells,
                num_ghost_cells_side,
                limiter
            );
    }
}
