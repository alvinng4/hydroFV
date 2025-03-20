/**
 * \file integrator_godunov_first_order.c
 * 
 * \brief First-order Godunov scheme for the Euler equations.
 * 
 * \author Ching-Yin Ng
 * \date 2025-03-20
 */

#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef USE_OPENMP
#include <omp.h>
#endif

#include "hydro.h"
#include "progress_bar.h"
#include "reconstruction.h"
#include "riemann_solver.h"
#include "source_term.h"
#include "utils.h"


IN_FILE void reconstruct_interface_1d(
    const IntegratorParam *__restrict integrator_param,
    const double *__restrict density,
    const double *__restrict velocity_x,
    const double *__restrict pressure,
    double *__restrict interface_density_L,
    double *__restrict interface_density_R,
    double *__restrict interface_velocity_x_L,
    double *__restrict interface_velocity_x_R,
    double *__restrict interface_pressure_L,
    double *__restrict interface_pressure_R,
    const int num_cells,
    const int num_ghost_cells_side
)
{
#ifdef USE_OPENMP
    #pragma omp parallel sections
    {
        #pragma omp section
        {
            reconstruct_cell_interface(
                integrator_param,
                density,
                interface_density_L,
                interface_density_R,
                num_cells,
                num_ghost_cells_side
            );
        }
        #pragma omp section
        {
            reconstruct_cell_interface(
                integrator_param,
                velocity_x,
                interface_velocity_x_L,
                interface_velocity_x_R,
                num_cells,
                num_ghost_cells_side
            );
        }
        #pragma omp section
        {
            reconstruct_cell_interface(
                integrator_param,
                pressure,
                interface_pressure_L,
                interface_pressure_R,
                num_cells,
                num_ghost_cells_side
            );
        }
    }
#else
    reconstruct_cell_interface(
        integrator_param,
        density,
        interface_density_L,
        interface_density_R,
        num_cells,
        num_ghost_cells_side
    );
    reconstruct_cell_interface(
        integrator_param,
        velocity_x,
        interface_velocity_x_L,
        interface_velocity_x_R,
        num_cells,
        num_ghost_cells_side
    );
    reconstruct_cell_interface(
        integrator_param,
        pressure,
        interface_pressure_L,
        interface_pressure_R,
        num_cells,
        num_ghost_cells_side
    );
#endif
}

IN_FILE ErrorStatus time_integrator_euler_1d(
    IntegratorParam *__restrict integrator_param,
    Settings *__restrict settings,
    double *__restrict density,
    double *__restrict velocity_x,
    double *__restrict pressure,
    double *__restrict mass,
    double *__restrict momentum_x,
    double *__restrict energy,
    double *__restrict volume,
    double *__restrict surface_area_x,
    const int num_cells,
    const int num_ghost_cells_side,
    const int num_interfaces,
    const double dt,
    const double gamma,
    const int boundary_condition_flag_x_min,
    const int boundary_condition_flag_x_max
)
{
    ErrorStatus error_status = make_success_error_status();
    
    double *__restrict interface_density_L = malloc(num_interfaces * sizeof(double));
    double *__restrict interface_density_R = malloc(num_interfaces * sizeof(double));
    double *__restrict interface_velocity_x_L = malloc(num_interfaces * sizeof(double));
    double *__restrict interface_velocity_x_R = malloc(num_interfaces * sizeof(double));
    double *__restrict interface_pressure_L = malloc(num_interfaces * sizeof(double));
    double *__restrict interface_pressure_R = malloc(num_interfaces * sizeof(double));
    if (
        !interface_density_L
        || !interface_density_R
        || !interface_velocity_x_L
        || !interface_velocity_x_R
        || !interface_pressure_L
        || !interface_pressure_R
    )
    {
        error_status = WRAP_RAISE_ERROR(MEMORY_ERROR, "Failed to allocate memory for interface arrays.");
        goto err_memory;
    }

    /* Reconstruct interface */
    reconstruct_interface_1d(
        integrator_param,
        density,
        velocity_x,
        pressure,
        interface_density_L,
        interface_density_R,
        interface_velocity_x_L,
        interface_velocity_x_R,
        interface_pressure_L,
        interface_pressure_R,
        num_cells,
        num_ghost_cells_side
    );

    /* Compute fluxes and update step */
#ifdef USE_OPENMP
    #pragma omp parallel for
#endif
    for (int i = 0; i < num_interfaces; i++)
    {
        double flux_mass;
        double flux_momentum_x;
        double flux_energy;
        ErrorStatus local_error_status = WRAP_TRACEBACK(solve_flux_1d(
            integrator_param,
            settings,
            &flux_mass,
            &flux_momentum_x,
            &flux_energy,
            gamma,
            interface_density_L[i],
            interface_velocity_x_L[i],
            interface_pressure_L[i],
            interface_density_R[i],
            interface_velocity_x_R[i],
            interface_pressure_R[i]
        ));
        if (local_error_status.return_code != SUCCESS)
        {
            error_status = local_error_status;
        }

        const double d_rho = flux_mass * dt;
        const double d_rho_u = flux_momentum_x * dt;
        const double d_energy_density = flux_energy * dt;

        const int idx = num_ghost_cells_side + i;
        mass[idx - 1] -= d_rho * surface_area_x[idx - 1];
        momentum_x[idx - 1] -= d_rho_u * surface_area_x[idx - 1];
        energy[idx - 1] -= d_energy_density * surface_area_x[idx - 1];

        mass[idx] += d_rho * surface_area_x[idx];
        momentum_x[idx] += d_rho_u * surface_area_x[idx];
        energy[idx] += d_energy_density * surface_area_x[idx];
    }

    convert_conserved_to_primitive_1d(
        num_cells,
        num_ghost_cells_side,
        gamma,
        volume,
        mass,
        momentum_x,
        energy,
        density,
        velocity_x,
        pressure
    );

    error_status = set_boundary_condition_1d(
        density,
        velocity_x,
        pressure,
        num_ghost_cells_side,
        num_cells,
        boundary_condition_flag_x_min,
        boundary_condition_flag_x_max
    );
    if (error_status.return_code != SUCCESS)
    {
        goto err_set_boundary_condition;
    }

    free(interface_density_L);
    free(interface_density_R);
    free(interface_velocity_x_L);
    free(interface_velocity_x_R);
    free(interface_pressure_L);
    free(interface_pressure_R);

    return error_status;

err_set_boundary_condition:
err_memory:
    free(interface_density_L);
    free(interface_density_R);
    free(interface_velocity_x_L);
    free(interface_velocity_x_R);
    free(interface_pressure_L);
    free(interface_pressure_R);

    return error_status;
}

IN_FILE ErrorStatus time_integrator_ssp_rk2_1d(
    IntegratorParam *__restrict integrator_param,
    Settings *__restrict settings,
    double *__restrict density,
    double *__restrict velocity_x,
    double *__restrict pressure,
    double *__restrict mass,
    double *__restrict momentum_x,
    double *__restrict energy,
    double *__restrict volume,
    double *__restrict surface_area_x,
    const int num_cells,
    const int num_ghost_cells_side,
    const int num_interfaces,
    const double dt,
    const double gamma,
    const int boundary_condition_flag_x_min,
    const int boundary_condition_flag_x_max
)
{
    ErrorStatus error_status = make_success_error_status();

    int stage;
    const int num_stages = 2;
    const int num_stages_minus_one = num_stages - 1;
    const double coeff[num_stages * 2] = {1.0, 1.0, 0.5, 0.5};

    const int total_num_cells = num_cells + 2 * num_ghost_cells_side;

    double *__restrict interface_density_L = malloc(num_interfaces * sizeof(double));
    double *__restrict interface_density_R = malloc(num_interfaces * sizeof(double));
    double *__restrict interface_velocity_x_L = malloc(num_interfaces * sizeof(double));
    double *__restrict interface_velocity_x_R = malloc(num_interfaces * sizeof(double));
    double *__restrict interface_pressure_L = malloc(num_interfaces * sizeof(double));
    double *__restrict interface_pressure_R = malloc(num_interfaces * sizeof(double));
    if (
        !interface_density_L
        || !interface_density_R
        || !interface_velocity_x_L
        || !interface_velocity_x_R
        || !interface_pressure_L
        || !interface_pressure_R
    )
    {
        error_status = WRAP_RAISE_ERROR(MEMORY_ERROR, "Failed to allocate memory for interface arrays.");
        goto err_memory_interface;
    }

    double *__restrict temp_density = malloc(total_num_cells * sizeof(double));
    double *__restrict temp_velocity_x = malloc(total_num_cells * sizeof(double));
    double *__restrict temp_pressure = malloc(total_num_cells * sizeof(double));

    double *__restrict temp_mass = malloc(num_stages_minus_one * total_num_cells * sizeof(double));
    double *__restrict temp_momentum_x = malloc(num_stages_minus_one * total_num_cells * sizeof(double));
    double *__restrict temp_energy = malloc(num_stages_minus_one * total_num_cells * sizeof(double));
    if (
        !temp_density
        || !temp_velocity_x
        || !temp_pressure
        || !temp_mass
        || !temp_momentum_x
        || !temp_energy
    )
    {
        error_status = WRAP_RAISE_ERROR(MEMORY_ERROR, "Failed to allocate memory for temporary arrays.");
        goto err_memory_temp;
    }

    /* Stage 0 (First stage) */
    stage = 0;

    memcpy(temp_mass, mass, total_num_cells * sizeof(double));
    memcpy(temp_momentum_x, momentum_x, total_num_cells * sizeof(double));
    memcpy(temp_energy, energy, total_num_cells * sizeof(double));

    /* Reconstruct interface */
    reconstruct_interface_1d(
        integrator_param,
        density,
        velocity_x,
        pressure,
        interface_density_L,
        interface_density_R,
        interface_velocity_x_L,
        interface_velocity_x_R,
        interface_pressure_L,
        interface_pressure_R,
        num_cells,
        num_ghost_cells_side
    );

    /* Compute fluxes and update step */
#ifdef USE_OPENMP
    #pragma omp parallel for
#endif
    for (int i = 0; i < num_interfaces; i++)
    {
        double flux_mass;
        double flux_momentum_x;
        double flux_energy;
        ErrorStatus local_error_status = WRAP_TRACEBACK(solve_flux_1d(
            integrator_param,
            settings,
            &flux_mass,
            &flux_momentum_x,
            &flux_energy,
            gamma,
            interface_density_L[i],
            interface_velocity_x_L[i],
            interface_pressure_L[i],
            interface_density_R[i],
            interface_velocity_x_R[i],
            interface_pressure_R[i]
        ));
        if (local_error_status.return_code != SUCCESS)
        {
            error_status = local_error_status;
        }

        const double d_rho = coeff[stage * 2 + 1] * flux_mass * dt;
        const double d_rho_u = coeff[stage * 2 + 1] * flux_momentum_x * dt;
        const double d_energy_density = coeff[stage * 2 + 1] * flux_energy * dt;

        const int idx = num_ghost_cells_side + i;
        temp_mass[stage * total_num_cells + (idx - 1)] -= d_rho * surface_area_x[idx - 1];
        temp_momentum_x[stage * total_num_cells + (idx - 1)] -= d_rho_u * surface_area_x[idx - 1];
        temp_energy[stage * total_num_cells + (idx - 1)] -= d_energy_density * surface_area_x[idx - 1];

        temp_mass[stage * total_num_cells + idx] += d_rho * surface_area_x[idx];
        temp_momentum_x[stage * total_num_cells + idx] += d_rho_u * surface_area_x[idx];
        temp_energy[stage * total_num_cells + idx] += d_energy_density * surface_area_x[idx];
    }

    convert_conserved_to_primitive_1d(
        num_cells,
        num_ghost_cells_side,
        gamma,
        volume,
        &(temp_mass[stage * total_num_cells]),
        &(temp_momentum_x[stage * total_num_cells]),
        &(temp_energy[stage * total_num_cells]),
        temp_density,
        temp_velocity_x,
        temp_pressure
    );

    error_status = set_boundary_condition_1d(
        temp_density,
        temp_velocity_x,
        temp_pressure,
        num_ghost_cells_side,
        num_cells,
        boundary_condition_flag_x_min,
        boundary_condition_flag_x_max
    );
    if (error_status.return_code != SUCCESS)
    {
        goto err_set_boundary_condition;
    }

    /* Stage 1 (Last stage) */
    stage = 1;

    /* Reconstruct interface */
    reconstruct_interface_1d(
        integrator_param,
        temp_density,
        temp_velocity_x,
        temp_pressure,
        interface_density_L,
        interface_density_R,
        interface_velocity_x_L,
        interface_velocity_x_R,
        interface_pressure_L,
        interface_pressure_R,
        num_cells,
        num_ghost_cells_side
    );

    /* Compute fluxes and update step */
#ifdef USE_OPENMP
    #pragma omp parallel for
#endif
    for (int i = num_ghost_cells_side; i < (num_ghost_cells_side + num_cells); i++)
    {
        mass[i] = coeff[stage * 2 + 0] * mass[i] + coeff[stage * 2 + 1] * temp_mass[(stage - 1) * total_num_cells + i];
        momentum_x[i] = coeff[stage * 2 + 0] * momentum_x[i] + coeff[stage * 2 + 1] * temp_momentum_x[(stage - 1) * total_num_cells + i];
        energy[i] = coeff[stage * 2 + 0] * energy[i] + coeff[stage * 2 + 1] * temp_energy[(stage - 1) * total_num_cells + i];
    }

#ifdef USE_OPENMP
    #pragma omp parallel for
#endif
    for (int i = 0; i < num_interfaces; i++)
    {
        double flux_mass;
        double flux_momentum_x;
        double flux_energy;
        ErrorStatus local_error_status = WRAP_TRACEBACK(solve_flux_1d(
            integrator_param,
            settings,
            &flux_mass,
            &flux_momentum_x,
            &flux_energy,
            gamma,
            interface_density_L[i],
            interface_velocity_x_L[i],
            interface_pressure_L[i],
            interface_density_R[i],
            interface_velocity_x_R[i],
            interface_pressure_R[i]
        ));
        if (local_error_status.return_code != SUCCESS)
        {
            error_status = local_error_status;
        }

        const double d_rho = coeff[stage * 2 + 1] * flux_mass * dt;
        const double d_rho_u = coeff[stage * 2 + 1] * flux_momentum_x * dt;
        const double d_energy_density = coeff[stage * 2 + 1] * flux_energy * dt;

        const int idx = num_ghost_cells_side + i;
        mass[idx - 1] -= d_rho * surface_area_x[idx - 1];
        momentum_x[idx - 1] -= d_rho_u * surface_area_x[idx - 1];
        energy[idx - 1] -= d_energy_density * surface_area_x[idx - 1];

        mass[idx] += d_rho * surface_area_x[idx];
        momentum_x[idx] += d_rho_u * surface_area_x[idx];
        energy[idx] += d_energy_density * surface_area_x[idx];
    }

    convert_conserved_to_primitive_1d(
        num_cells,
        num_ghost_cells_side,
        gamma,
        volume,
        mass,
        momentum_x,
        energy,
        density,
        velocity_x,
        pressure
    );

    error_status = set_boundary_condition_1d(
        density,
        velocity_x,
        pressure,
        num_ghost_cells_side,
        num_cells,
        boundary_condition_flag_x_min,
        boundary_condition_flag_x_max
    );
    if (error_status.return_code != SUCCESS)
    {
        goto err_set_boundary_condition;
    }

    free(interface_density_L);
    free(interface_density_R);
    free(interface_velocity_x_L);
    free(interface_velocity_x_R);
    free(interface_pressure_L);
    free(interface_pressure_R);

    free(temp_density);
    free(temp_velocity_x);
    free(temp_pressure);
    free(temp_mass);
    free(temp_momentum_x);
    free(temp_energy);

    return error_status;

err_set_boundary_condition:
err_memory_temp:
    free(temp_density);
    free(temp_velocity_x);
    free(temp_pressure);
    free(temp_mass);
    free(temp_momentum_x);
    free(temp_energy);
err_memory_interface:
    free(interface_density_L);
    free(interface_density_R);
    free(interface_velocity_x_L);
    free(interface_velocity_x_R);
    free(interface_pressure_L);
    free(interface_pressure_R);

    return error_status;
}

IN_FILE ErrorStatus time_integrator_ssp_rk3_1d(
    IntegratorParam *__restrict integrator_param,
    Settings *__restrict settings,
    double *__restrict density,
    double *__restrict velocity_x,
    double *__restrict pressure,
    double *__restrict mass,
    double *__restrict momentum_x,
    double *__restrict energy,
    double *__restrict volume,
    double *__restrict surface_area_x,
    const int num_cells,
    const int num_ghost_cells_side,
    const int num_interfaces,
    const double dt,
    const double gamma,
    const int boundary_condition_flag_x_min,
    const int boundary_condition_flag_x_max
)
{
    ErrorStatus error_status = make_success_error_status();

    int stage;
    const int num_stages = 3;
    const int num_stages_minus_one = num_stages - 1;
    const double coeff[num_stages * 2] = {
        1.0, 1.0,
        0.75, 0.25,
        1.0 / 3.0, 2.0 / 3.0
    };

    const int total_num_cells = num_cells + 2 * num_ghost_cells_side;

    double *__restrict interface_density_L = malloc(num_interfaces * sizeof(double));
    double *__restrict interface_density_R = malloc(num_interfaces * sizeof(double));
    double *__restrict interface_velocity_x_L = malloc(num_interfaces * sizeof(double));
    double *__restrict interface_velocity_x_R = malloc(num_interfaces * sizeof(double));
    double *__restrict interface_pressure_L = malloc(num_interfaces * sizeof(double));
    double *__restrict interface_pressure_R = malloc(num_interfaces * sizeof(double));
    if (
        !interface_density_L
        || !interface_density_R
        || !interface_velocity_x_L
        || !interface_velocity_x_R
        || !interface_pressure_L
        || !interface_pressure_R
    )
    {
        error_status = WRAP_RAISE_ERROR(MEMORY_ERROR, "Failed to allocate memory for interface arrays.");
        goto err_memory_interface;
    }

    double *__restrict temp_density = malloc(total_num_cells * sizeof(double));
    double *__restrict temp_velocity_x = malloc(total_num_cells * sizeof(double));
    double *__restrict temp_pressure = malloc(total_num_cells * sizeof(double));

    double *__restrict temp_mass = malloc(num_stages_minus_one * total_num_cells * sizeof(double));
    double *__restrict temp_momentum_x = malloc(num_stages_minus_one * total_num_cells * sizeof(double));
    double *__restrict temp_energy = malloc(num_stages_minus_one * total_num_cells * sizeof(double));
    if (
        !temp_density
        || !temp_velocity_x
        || !temp_pressure
        || !temp_mass
        || !temp_momentum_x
        || !temp_energy
    )
    {
        error_status = WRAP_RAISE_ERROR(MEMORY_ERROR, "Failed to allocate memory for temporary arrays.");
        goto err_memory_temp;
    }

    /* Stage 0 (First stage) */
    stage = 0;

    memcpy(temp_mass, mass, total_num_cells * sizeof(double));
    memcpy(temp_momentum_x, momentum_x, total_num_cells * sizeof(double));
    memcpy(temp_energy, energy, total_num_cells * sizeof(double));

    /* Reconstruct interface */
    reconstruct_interface_1d(
        integrator_param,
        density,
        velocity_x,
        pressure,
        interface_density_L,
        interface_density_R,
        interface_velocity_x_L,
        interface_velocity_x_R,
        interface_pressure_L,
        interface_pressure_R,
        num_cells,
        num_ghost_cells_side
    );

    /* Compute fluxes and update step */
#ifdef USE_OPENMP
    #pragma omp parallel for
#endif
    for (int i = 0; i < num_interfaces; i++)
    {
        double flux_mass;
        double flux_momentum_x;
        double flux_energy;
        ErrorStatus local_error_status = WRAP_TRACEBACK(solve_flux_1d(
            integrator_param,
            settings,
            &flux_mass,
            &flux_momentum_x,
            &flux_energy,
            gamma,
            interface_density_L[i],
            interface_velocity_x_L[i],
            interface_pressure_L[i],
            interface_density_R[i],
            interface_velocity_x_R[i],
            interface_pressure_R[i]
        ));
        if (local_error_status.return_code != SUCCESS)
        {
            error_status = local_error_status;
        }

        const double d_rho = coeff[stage * 2 + 1] * flux_mass * dt;
        const double d_rho_u = coeff[stage * 2 + 1] * flux_momentum_x * dt;
        const double d_energy_density = coeff[stage * 2 + 1] * flux_energy * dt;

        const int idx = num_ghost_cells_side + i;
        temp_mass[stage * total_num_cells + (idx - 1)] -= d_rho * surface_area_x[idx - 1];
        temp_momentum_x[stage * total_num_cells + (idx - 1)] -= d_rho_u * surface_area_x[idx - 1];
        temp_energy[stage * total_num_cells + (idx - 1)] -= d_energy_density * surface_area_x[idx - 1];

        temp_mass[stage * total_num_cells + idx] += d_rho * surface_area_x[idx];
        temp_momentum_x[stage * total_num_cells + idx] += d_rho_u * surface_area_x[idx];
        temp_energy[stage * total_num_cells + idx] += d_energy_density * surface_area_x[idx];
    }

    convert_conserved_to_primitive_1d(
        num_cells,
        num_ghost_cells_side,
        gamma,
        volume,
        &(temp_mass[stage * total_num_cells]),
        &(temp_momentum_x[stage * total_num_cells]),
        &(temp_energy[stage * total_num_cells]),
        temp_density,
        temp_velocity_x,
        temp_pressure
    );

    error_status = set_boundary_condition_1d(
        temp_density,
        temp_velocity_x,
        temp_pressure,
        num_ghost_cells_side,
        num_cells,
        boundary_condition_flag_x_min,
        boundary_condition_flag_x_max
    );
    if (error_status.return_code != SUCCESS)
    {
        goto err_set_boundary_condition;
    }

    /* Stage 1 */
    stage = 1;

#ifdef USE_OPENMP
    #pragma omp parallel for
#endif
    for (int i = num_ghost_cells_side; i < (num_ghost_cells_side + num_cells); i++)
    {
        temp_mass[stage * total_num_cells + i] = coeff[stage * 2 + 0] * mass[i] + coeff[stage * 2 + 1] * temp_mass[(stage - 1) * total_num_cells + i];
        temp_momentum_x[stage * total_num_cells + i] = coeff[stage * 2 + 0] * momentum_x[i] + coeff[stage * 2 + 1] * temp_momentum_x[(stage - 1) * total_num_cells + i];
        temp_energy[stage * total_num_cells + i] = coeff[stage * 2 + 0] * energy[i] + coeff[stage * 2 + 1] * temp_energy[(stage - 1) * total_num_cells + i];
    }

    /* Reconstruct interface */
    reconstruct_interface_1d(
        integrator_param,
        temp_density,
        temp_velocity_x,
        temp_pressure,
        interface_density_L,
        interface_density_R,
        interface_velocity_x_L,
        interface_velocity_x_R,
        interface_pressure_L,
        interface_pressure_R,
        num_cells,
        num_ghost_cells_side
    );

    /* Compute fluxes and update step */
#ifdef USE_OPENMP
    #pragma omp parallel for
#endif
    for (int i = 0; i < num_interfaces; i++)
    {
        double flux_mass;
        double flux_momentum_x;
        double flux_energy;
        ErrorStatus local_error_status = WRAP_TRACEBACK(solve_flux_1d(
            integrator_param,
            settings,
            &flux_mass,
            &flux_momentum_x,
            &flux_energy,
            gamma,
            interface_density_L[i],
            interface_velocity_x_L[i],
            interface_pressure_L[i],
            interface_density_R[i],
            interface_velocity_x_R[i],
            interface_pressure_R[i]
        ));
        if (local_error_status.return_code != SUCCESS)
        {
            error_status = local_error_status;
        }

        const double d_rho = coeff[stage * 2 + 1] * flux_mass * dt;
        const double d_rho_u = coeff[stage * 2 + 1] * flux_momentum_x * dt;
        const double d_energy_density = coeff[stage * 2 + 1] * flux_energy * dt;

        const int idx = num_ghost_cells_side + i;
        temp_mass[stage * total_num_cells + (idx - 1)] -= d_rho * surface_area_x[idx - 1];
        temp_momentum_x[stage * total_num_cells + (idx - 1)] -= d_rho_u * surface_area_x[idx - 1];
        temp_energy[stage * total_num_cells + (idx - 1)] -= d_energy_density * surface_area_x[idx - 1];

        temp_mass[stage * total_num_cells + idx] += d_rho * surface_area_x[idx];
        temp_momentum_x[stage * total_num_cells + idx] += d_rho_u * surface_area_x[idx];
        temp_energy[stage * total_num_cells + idx] += d_energy_density * surface_area_x[idx];
    }

    convert_conserved_to_primitive_1d(
        num_cells,
        num_ghost_cells_side,
        gamma,
        volume,
        &(temp_mass[stage * total_num_cells]),
        &(temp_momentum_x[stage * total_num_cells]),
        &(temp_energy[stage * total_num_cells]),
        temp_density,
        temp_velocity_x,
        temp_pressure
    );

    error_status = set_boundary_condition_1d(
        temp_density,
        temp_velocity_x,
        temp_pressure,
        num_ghost_cells_side,
        num_cells,
        boundary_condition_flag_x_min,
        boundary_condition_flag_x_max
    );
    if (error_status.return_code != SUCCESS)
    {
        goto err_set_boundary_condition;
    }

    /* Stage 2 (Last stage) */
    stage = 2;

    /* Reconstruct interface */
    reconstruct_interface_1d(
        integrator_param,
        temp_density,
        temp_velocity_x,
        temp_pressure,
        interface_density_L,
        interface_density_R,
        interface_velocity_x_L,
        interface_velocity_x_R,
        interface_pressure_L,
        interface_pressure_R,
        num_cells,
        num_ghost_cells_side
    );

    /* Compute fluxes and update step */
#ifdef USE_OPENMP
    #pragma omp parallel for
#endif
    for (int i = num_ghost_cells_side; i < (num_ghost_cells_side + num_cells); i++)
    {
        mass[i] = coeff[stage * 2 + 0] * mass[i] + coeff[stage * 2 + 1] * temp_mass[(stage - 1) * total_num_cells + i];
        momentum_x[i] = coeff[stage * 2 + 0] * momentum_x[i] + coeff[stage * 2 + 1] * temp_momentum_x[(stage - 1) * total_num_cells + i];
        energy[i] = coeff[stage * 2 + 0] * energy[i] + coeff[stage * 2 + 1] * temp_energy[(stage - 1) * total_num_cells + i];
    }

#ifdef USE_OPENMP
    #pragma omp parallel for
#endif
    for (int i = 0; i < num_interfaces; i++)
    {
        double flux_mass;
        double flux_momentum_x;
        double flux_energy;
        ErrorStatus local_error_status = WRAP_TRACEBACK(solve_flux_1d(
            integrator_param,
            settings,
            &flux_mass,
            &flux_momentum_x,
            &flux_energy,
            gamma,
            interface_density_L[i],
            interface_velocity_x_L[i],
            interface_pressure_L[i],
            interface_density_R[i],
            interface_velocity_x_R[i],
            interface_pressure_R[i]
        ));
        if (local_error_status.return_code != SUCCESS)
        {
            error_status = local_error_status;
        }

        const double d_rho = coeff[stage * 2 + 1] * flux_mass * dt;
        const double d_rho_u = coeff[stage * 2 + 1] * flux_momentum_x * dt;
        const double d_energy_density = coeff[stage * 2 + 1] * flux_energy * dt;

        const int idx = num_ghost_cells_side + i;
        mass[idx - 1] -= d_rho * surface_area_x[idx - 1];
        momentum_x[idx - 1] -= d_rho_u * surface_area_x[idx - 1];
        energy[idx - 1] -= d_energy_density * surface_area_x[idx - 1];

        mass[idx] += d_rho * surface_area_x[idx];
        momentum_x[idx] += d_rho_u * surface_area_x[idx];
        energy[idx] += d_energy_density * surface_area_x[idx];
    }

    convert_conserved_to_primitive_1d(
        num_cells,
        num_ghost_cells_side,
        gamma,
        volume,
        mass,
        momentum_x,
        energy,
        density,
        velocity_x,
        pressure
    );

    error_status = set_boundary_condition_1d(
        density,
        velocity_x,
        pressure,
        num_ghost_cells_side,
        num_cells,
        boundary_condition_flag_x_min,
        boundary_condition_flag_x_max
    );
    if (error_status.return_code != SUCCESS)
    {
        goto err_set_boundary_condition;
    }

    free(interface_density_L);
    free(interface_density_R);
    free(interface_velocity_x_L);
    free(interface_velocity_x_R);
    free(interface_pressure_L);
    free(interface_pressure_R);

    free(temp_density);
    free(temp_velocity_x);
    free(temp_pressure);
    free(temp_mass);
    free(temp_momentum_x);
    free(temp_energy);

    return error_status;

err_set_boundary_condition:
err_memory_temp:
    free(temp_density);
    free(temp_velocity_x);
    free(temp_pressure);
    free(temp_mass);
    free(temp_momentum_x);
    free(temp_energy);
err_memory_interface:
    free(interface_density_L);
    free(interface_density_R);
    free(interface_velocity_x_L);
    free(interface_velocity_x_R);
    free(interface_pressure_L);
    free(interface_pressure_R);

    return error_status;
}

ErrorStatus godunov_first_order_1d(
    System *__restrict system,
    IntegratorParam *__restrict integrator_param,
    StoringParam *__restrict storing_param,
    Settings *__restrict settings,
    SimulationParam *__restrict simulation_param,
    SimulationStatus *__restrict simulation_status
)
{
    /* Declare variables */
    ErrorStatus error_status = make_success_error_status();

    bool is_compute_geometry_source_term = true;
    if (system->coord_sys_flag_ == COORD_SYS_CARTESIAN_1D)
    {
        is_compute_geometry_source_term = false;
    }

    /* System parameters */
    const double gamma = system->gamma;
    const int num_cells = system->num_cells_x;
    const int num_ghost_cells_side = system->num_ghost_cells_side;
    const int num_interfaces = num_cells + 1;
    const int boundary_condition_flag_x_min = system->boundary_condition_flag_x_min_;
    const int boundary_condition_flag_x_max = system->boundary_condition_flag_x_max_;

    /* Integrator parameters */
    const double cfl = integrator_param->cfl;
    const double cfl_initial_shrink_factor = integrator_param->cfl_initial_shrink_factor;
    const int cfl_initial_shrink_num_steps = integrator_param->cfl_initial_shrink_num_steps;

    /* Storing parameters */
    const bool is_storing = storing_param->is_storing;
    const double storing_interval = storing_param->storing_interval;
    int *__restrict store_count_ptr = &storing_param->store_count_;

    /* Settings */
    const bool no_progress_bar = settings->no_progress_bar;

    /* Simulation parameters */
    const double tf = simulation_param->tf;

    /* Simulation status */
    int64 *__restrict num_steps_ptr = &simulation_status->num_steps;
    double *__restrict dt_ptr = &simulation_status->dt;
    double *__restrict t_ptr = &simulation_status->t;

    /* Arrays */
    double *__restrict density = system->density_;
    double *__restrict velocity_x = system->velocity_x_;
    double *__restrict pressure = system->pressure_;
    double *__restrict mass = system->mass_;
    double *__restrict momentum_x = system->momentum_x_;
    double *__restrict energy = system->energy_;
    double *__restrict surface_area_x = system->surface_area_x_;

    /* Time integrator function */
    ErrorStatus (*time_integrator_1d_func)(
        IntegratorParam *__restrict integrator_param,
        Settings *__restrict settings,
        double *__restrict density,
        double *__restrict velocity_x,
        double *__restrict pressure,
        double *__restrict mass,
        double *__restrict momentum_x,
        double *__restrict energy,
        double *__restrict volume,
        double *__restrict surface_area_x,
        const int num_cells,
        const int num_ghost_cells_side,
        const int num_interfaces,
        const double dt,
        const double gamma,
        const int boundary_condition_flag_x_min,
        const int boundary_condition_flag_x_max
    );

    switch (integrator_param->time_integrator_flag_)
    {
        case TIME_INTEGRATOR_EULER:
            time_integrator_1d_func = time_integrator_euler_1d;
            break;
        case TIME_INTEGRATOR_SSP_RK2:
            time_integrator_1d_func = time_integrator_ssp_rk2_1d;
            break;
        case TIME_INTEGRATOR_SSP_RK3:
            time_integrator_1d_func = time_integrator_ssp_rk3_1d;
            break;
        default:
            error_status = WRAP_RAISE_ERROR(VALUE_ERROR, "Time integrator not recognized.");
            return error_status;
    }

    /* Main Loop */
    ProgressBarParam progress_bar_param;
    if (!no_progress_bar)
    {
        start_progress_bar(&progress_bar_param, tf);
    }

    *num_steps_ptr = 0;
    *dt_ptr = 0.0;
    *t_ptr = 0.0;

    /* Store first snapshot */
    *store_count_ptr = 0;
    const int store_initial_offset = (is_storing && storing_param->store_initial) ? 1 : 0;
    if (is_storing && storing_param->store_initial)
    {
        error_status = WRAP_TRACEBACK(store_snapshot(system, integrator_param, simulation_status, storing_param));
        if (error_status.return_code != SUCCESS)
        {
            goto err_store_first_snapshot;
        }
    }

    while (*t_ptr < tf)
    {
        /* Compute time step */
        if (*num_steps_ptr < cfl_initial_shrink_num_steps)
        {
            *dt_ptr = get_time_step_1d(system, cfl * cfl_initial_shrink_factor);
        }
        else
        {
            *dt_ptr = get_time_step_1d(system, cfl);
        }
        if (*t_ptr + (*dt_ptr) > tf)
        {
            *dt_ptr = tf - *t_ptr;
        }
        if ((*dt_ptr) == 0.0)
        {
            error_status = WRAP_RAISE_ERROR(VALUE_ERROR, "dt is zero.");
            goto err_dt_zero;
        }
        const double dt = (*dt_ptr);

        /* Advance step */
        error_status = time_integrator_1d_func(
            integrator_param,
            settings,
            density,
            velocity_x,
            pressure,
            mass,
            momentum_x,
            energy,
            system->volume_,
            surface_area_x,
            num_cells,
            num_ghost_cells_side,
            num_interfaces,
            dt,
            gamma,
            boundary_condition_flag_x_min,
            boundary_condition_flag_x_max
        );
        if (error_status.return_code != SUCCESS)
        {
            goto err_advance_step;
        }

        if (is_compute_geometry_source_term)
        {
            error_status = WRAP_TRACEBACK(add_geometry_source_term(system, dt));
            if (error_status.return_code != SUCCESS)
            {
                goto err_compute_geometry_source_term;
            }
        }

        (*t_ptr) += dt;
        (*num_steps_ptr)++;

        /* Store snapshot */
        if (is_storing && ((*t_ptr) >= storing_interval * (*store_count_ptr - store_initial_offset + 1)))
        {
            error_status = WRAP_TRACEBACK(store_snapshot(system, integrator_param, simulation_status, storing_param));
            if (error_status.return_code != SUCCESS)
            {
                goto err_store_snapshot;
            }
        }

        if (!no_progress_bar)
        {
            update_progress_bar(&progress_bar_param, *t_ptr, false);
        }
    }

    if (!no_progress_bar)
    {
        update_progress_bar(&progress_bar_param, *t_ptr, true);
    }

    return make_success_error_status();

err_store_snapshot:
err_compute_geometry_source_term:
err_advance_step:
err_dt_zero:
    if (!no_progress_bar)
    {
        update_progress_bar(&progress_bar_param, *t_ptr, true);
    }
err_store_first_snapshot:

    return error_status;
}

// IN_FILE void reconstruct_interface_2d(
//     const IntegratorParam *__restrict integrator_param,
//     const double *__restrict density,
//     const double *__restrict velocity_x,
//     const double *__restrict velocity_y,
//     const double *__restrict pressure,
//     double *__restrict interface_density_L,
//     double *__restrict interface_density_R,
//     double *__restrict interface_velocity_x_x_L,
//     double *__restrict interface_velocity_x_x_R,
//     double *__restrict interface_velocity_x_y_L,
//     double *__restrict interface_velocity_x_y_R,
//     double *__restrict interface_pressure_L,
//     double *__restrict interface_pressure_R,
//     const int num_cells_x,
//     const int num_ghost_cells_side
// )
// {
//     reconstruct_cell_interface(
//         integrator_param,
//         density,
//         interface_density_L,
//         interface_density_R,
//         num_cells_x,
//         num_ghost_cells_side
//     );
//     reconstruct_cell_interface(
//         integrator_param,
//         velocity_x,
//         interface_velocity_x_x_L,
//         interface_velocity_x_x_R,
//         num_cells_x,
//         num_ghost_cells_side
//     );
//     reconstruct_cell_interface(
//         integrator_param,
//         velocity_y,
//         interface_velocity_x_y_L,
//         interface_velocity_x_y_R,
//         num_cells_x,
//         num_ghost_cells_side
//     );
//     reconstruct_cell_interface(
//         integrator_param,
//         pressure,
//         interface_pressure_L,
//         interface_pressure_R,
//         num_cells_x,
//         num_ghost_cells_side
//     );
// }


// ErrorStatus godunov_first_order_2d(
//     System *__restrict system,
//     IntegratorParam *__restrict integrator_param,
//     StoringParam *__restrict storing_param,
//     Settings *__restrict settings,
//     SimulationParam *__restrict simulation_param,
//     SimulationStatus *__restrict simulation_status
// )
// {
//     /* Declare variables */
//     ErrorStatus error_status = make_success_error_status();

//     /* System parameters */
//     const double gamma = system->gamma;
//     const int num_ghost_cells_side = system->num_ghost_cells_side;
//     const int num_cells_x = system->num_cells_x;
//     const int num_cells_y = system->num_cells_y;
//     const int total_num_cells_x = num_cells_x + 2 * num_ghost_cells_side;
//     const int total_num_cells_y = num_cells_y + 2 * num_ghost_cells_side;
//     const int num_interfaces_x = num_cells_x + 1;
//     const int num_interfaces_y = num_cells_y + 1;

//     /* Integrator parameters */
//     const double cfl = integrator_param->cfl;
//     const double cfl_initial_shrink_factor = integrator_param->cfl_initial_shrink_factor;
//     const int cfl_initial_shrink_num_steps = integrator_param->cfl_initial_shrink_num_steps;

//     /* Storing parameters */
//     const bool is_storing = storing_param->is_storing;
//     const double storing_interval = storing_param->storing_interval;
//     int *__restrict store_count_ptr = &storing_param->store_count_;

//     /* Settings */
//     const bool no_progress_bar = settings->no_progress_bar;

//     /* Simulation parameters */
//     const double tf = simulation_param->tf;

//     /* Simulation status */
//     int64 *__restrict num_steps_ptr = &simulation_status->num_steps;
//     double *__restrict dt_ptr = &simulation_status->dt;
//     double *__restrict t_ptr = &simulation_status->t;

//     /* Arrays */
//     double *__restrict density = system->density_;
//     double *__restrict velocity_x = system->velocity_x_;
//     double *__restrict velocity_y = system->velocity_y_;
//     double *__restrict pressure = system->pressure_;
//     double *__restrict mass = system->mass_;
//     double *__restrict momentum_x = system->momentum_x_;
//     double *__restrict momentum_y = system->momentum_y_;
//     double *__restrict energy = system->energy_;
//     double *__restrict surface_area_x = system->surface_area_x_;
//     double *__restrict surface_area_y = system->surface_area_y_;

//     double *__restrict temp_density_y = malloc(total_num_cells_y * sizeof(double));
//     double *__restrict temp_velocity_y_x = malloc(total_num_cells_y * sizeof(double));
//     double *__restrict temp_velocity_y_y = malloc(total_num_cells_y * sizeof(double));
//     double *__restrict temp_pressure_y = malloc(total_num_cells_y * sizeof(double));
//     if (
//         !temp_density_y
//         || !temp_velocity_y_x
//         || !temp_velocity_y_y
//         || !temp_pressure_y
//     )
//     {
//         error_status = WRAP_RAISE_ERROR(MEMORY_ERROR, "Failed to allocate memory for temporary arrays.");
//         goto err_allocate_temp_arrays;
//     }

//     double *__restrict interface_density_x_L = malloc(num_interfaces_x * sizeof(double));
//     double *__restrict interface_density_x_R = malloc(num_interfaces_x * sizeof(double));
//     double *__restrict interface_velocity_x_x_L = malloc(num_interfaces_x * sizeof(double));
//     double *__restrict interface_velocity_x_x_R = malloc(num_interfaces_x * sizeof(double));
//     double *__restrict interface_velocity_x_y_L = malloc(num_interfaces_x * sizeof(double));
//     double *__restrict interface_velocity_x_y_R = malloc(num_interfaces_x * sizeof(double));
//     double *__restrict interface_pressure_x_L = malloc(num_interfaces_x * sizeof(double));
//     double *__restrict interface_pressure_x_R = malloc(num_interfaces_x * sizeof(double));

//     double *__restrict interface_density_y_B = malloc(num_interfaces_y * sizeof(double));
//     double *__restrict interface_density_y_T = malloc(num_interfaces_y * sizeof(double));
//     double *__restrict interface_velocity_y_x_B = malloc(num_interfaces_y * sizeof(double));
//     double *__restrict interface_velocity_y_x_T = malloc(num_interfaces_y * sizeof(double));
//     double *__restrict interface_velocity_y_y_B = malloc(num_interfaces_y * sizeof(double));
//     double *__restrict interface_velocity_y_y_T = malloc(num_interfaces_y * sizeof(double));
//     double *__restrict interface_pressure_y_B = malloc(num_interfaces_y * sizeof(double));
//     double *__restrict interface_pressure_y_T = malloc(num_interfaces_y * sizeof(double));

//     if (
//         !interface_density_x_L
//         || !interface_density_x_R
//         || !interface_velocity_x_x_L
//         || !interface_velocity_x_x_R
//         || !interface_velocity_x_y_L
//         || !interface_velocity_x_y_R
//         || !interface_pressure_x_L
//         || !interface_pressure_x_R
//         || !interface_density_y_B
//         || !interface_density_y_T
//         || !interface_velocity_y_x_B
//         || !interface_velocity_y_x_T
//         || !interface_velocity_y_y_B
//         || !interface_velocity_y_y_T
//         || !interface_pressure_y_B
//         || !interface_pressure_y_T
//     )
//     {
//         error_status = WRAP_RAISE_ERROR(MEMORY_ERROR, "Failed to allocate memory for interface arrays.");
//         goto err_allocate_interface_arrays;
//     }

//     /* Main Loop */
//     ProgressBarParam progress_bar_param;
//     if (!no_progress_bar)
//     {
//         start_progress_bar(&progress_bar_param, tf);
//     }

//     *num_steps_ptr = 0;
//     *dt_ptr = 0.0;
//     *t_ptr = 0.0;

//     /* Store first snapshot */
//     *store_count_ptr = 0;
//     const int store_initial_offset = (is_storing && storing_param->store_initial) ? 1 : 0;
//     if (is_storing && storing_param->store_initial)
//     {
//         error_status = WRAP_TRACEBACK(store_snapshot(system, integrator_param, simulation_status, storing_param));
//         if (error_status.return_code != SUCCESS)
//         {
//             goto err_store_first_snapshot;
//         }
//     }

//     while (*t_ptr < tf)
//     {
//         /* Compute time step */
//         if (*num_steps_ptr < cfl_initial_shrink_num_steps)
//         {
//             *dt_ptr = get_time_step_2d(system, cfl * cfl_initial_shrink_factor);
//         }
//         else
//         {
//             *dt_ptr = get_time_step_2d(system, cfl);
//         }
//         if (*t_ptr + (*dt_ptr) > tf)
//         {
//             *dt_ptr = tf - *t_ptr;
//         }
//         if ((*dt_ptr) == 0.0)
//         {
//             error_status = WRAP_RAISE_ERROR(VALUE_ERROR, "dt is zero.");
//             goto err_dt_zero;
//         }
//         const double dt = (*dt_ptr);

//         /* Compute fluxes and update step (x-direction) */
// #ifdef USE_OPENMP
//         #pragma omp parallel for
// #endif
//         for (int j = num_ghost_cells_side; j < (num_ghost_cells_side + num_cells_y); j++)
//         {
//             // Reconstruct interface
//             reconstruct_interface_2d(
//                 integrator_param,
//                 &(density[j * total_num_cells_x]),
//                 &(velocity_x[j * total_num_cells_x]),
//                 &(velocity_y[j * total_num_cells_x]),
//                 &(pressure[j * total_num_cells_x]),
//                 interface_density_x_L,
//                 interface_density_x_R,
//                 interface_velocity_x_x_L,
//                 interface_velocity_x_x_R,
//                 interface_velocity_x_y_L,
//                 interface_velocity_x_y_R,
//                 interface_pressure_x_L,
//                 interface_pressure_x_R,
//                 num_cells_x,
//                 num_ghost_cells_side
//             );

//             // Compute fluxes and update step
//             for (int i = 0; i < num_interfaces_x; i++)
//             {
//                 ErrorStatus local_error_status;

//                 /* x-direction */
//                 double flux_mass_x;
//                 double flux_momentum_x_x;
//                 double flux_momentum_x_y;
//                 double flux_energy_x;

//                 local_error_status = WRAP_TRACEBACK(solve_flux_2d(
//                     integrator_param,
//                     settings,
//                     &flux_mass_x,
//                     &flux_momentum_x_x,
//                     &flux_momentum_x_y,
//                     &flux_energy_x,
//                     gamma,
//                     interface_density_x_L[i],
//                     interface_velocity_x_x_L[i],
//                     interface_velocity_x_y_L[i],
//                     interface_pressure_x_L[i],
//                     interface_density_x_R[i],
//                     interface_velocity_x_x_R[i],
//                     interface_velocity_x_y_R[i],
//                     interface_pressure_x_R[i]
//                 ));
//                 if (local_error_status.return_code != SUCCESS)
//                 {
//                     error_status = local_error_status;
//                 }

//                 const int idx_i = num_ghost_cells_side + i;
//                 mass[j * total_num_cells_x + (idx_i - 1)] -= dt * flux_mass_x * surface_area_x[idx_i - 1];
//                 momentum_x[j * total_num_cells_x + (idx_i - 1)] -= dt * flux_momentum_x_x * surface_area_x[idx_i - 1];
//                 momentum_y[j * total_num_cells_x + (idx_i - 1)] -= dt * flux_momentum_x_y * surface_area_x[idx_i - 1];
//                 energy[j * total_num_cells_x + (idx_i - 1)] -= dt * flux_energy_x * surface_area_x[idx_i - 1];

//                 mass[j * total_num_cells_x + idx_i] += dt * flux_mass_x * surface_area_x[idx_i];
//                 momentum_x[j * total_num_cells_x + idx_i] += dt * flux_momentum_x_x * surface_area_x[idx_i];
//                 momentum_y[j * total_num_cells_x + idx_i] += dt * flux_momentum_x_y * surface_area_x[idx_i];
//                 energy[j * total_num_cells_x + idx_i] += dt * flux_energy_x * surface_area_x[idx_i];
//             }
//         }
//         if (error_status.return_code != SUCCESS)
//         {
//             goto err_solve_flux;
//         }

//         /* Compute fluxes and update step (y-direction) */
// #ifdef USE_OPENMP
//         #pragma omp parallel for
// #endif
//         for (int i = num_ghost_cells_side; i < (num_ghost_cells_side + num_cells_x); i++)
//         {
//             // Reconstruct interface
//             for (int j = 0; j < total_num_cells_y; j++)
//             {
//                 temp_density_y[j] = density[j * total_num_cells_x + i];
//                 temp_velocity_y_x[j] = velocity_x[j * total_num_cells_x + i];
//                 temp_velocity_y_y[j] = velocity_y[j * total_num_cells_x + i];
//                 temp_pressure_y[j] = pressure[j * total_num_cells_x + i];
//             }
//             reconstruct_interface_2d(
//                 integrator_param,
//                 temp_density_y,
//                 temp_velocity_y_y,
//                 temp_velocity_y_x,
//                 temp_pressure_y,
//                 interface_density_y_B,
//                 interface_density_y_T,
//                 interface_velocity_y_y_B,
//                 interface_velocity_y_y_T,
//                 interface_velocity_y_x_B,
//                 interface_velocity_y_x_T,
//                 interface_pressure_y_B,
//                 interface_pressure_y_T,
//                 num_cells_y,
//                 num_ghost_cells_side
//             );

//             // Compute fluxes and update step
//             for (int j = 0; j < num_interfaces_y; j++)
//             {
//                 ErrorStatus local_error_status;

//                 /* y-direction */
//                 double flux_mass_y;
//                 double flux_momentum_y_x;
//                 double flux_momentum_y_y;
//                 double flux_energy_y;

//                 local_error_status = WRAP_TRACEBACK(solve_flux_2d(
//                     integrator_param,
//                     settings,
//                     &flux_mass_y,
//                     &flux_momentum_y_y,
//                     &flux_momentum_y_x,
//                     &flux_energy_y,
//                     gamma,
//                     interface_density_y_B[j],
//                     interface_velocity_y_y_B[j],
//                     interface_velocity_y_x_B[j],
//                     interface_pressure_y_B[j],
//                     interface_density_y_T[j],
//                     interface_velocity_y_y_T[j],
//                     interface_velocity_y_x_T[j],
//                     interface_pressure_y_T[j]
//                 ));
//                 if (local_error_status.return_code != SUCCESS)
//                 {
//                     error_status = local_error_status;
//                 }

//                 const int idx_j = num_ghost_cells_side + j;

//                 mass[(idx_j - 1) * total_num_cells_x + i] -= dt * flux_mass_y * surface_area_y[idx_j - 1];
//                 momentum_x[(idx_j - 1) * total_num_cells_x + i] -= dt * flux_momentum_y_x * surface_area_y[idx_j - 1];
//                 momentum_y[(idx_j - 1) * total_num_cells_x + i] -= dt * flux_momentum_y_y * surface_area_y[idx_j - 1];
//                 energy[(idx_j - 1) * total_num_cells_x + i] -= dt * flux_energy_y * surface_area_y[idx_j - 1];

//                 mass[idx_j * total_num_cells_x + i] += dt * flux_mass_y * surface_area_y[idx_j];
//                 momentum_x[idx_j * total_num_cells_x + i] += dt * flux_momentum_y_x * surface_area_y[idx_j];
//                 momentum_y[idx_j * total_num_cells_x + i] += dt * flux_momentum_y_y * surface_area_y[idx_j];
//                 energy[idx_j * total_num_cells_x + i] += dt * flux_energy_y * surface_area_y[idx_j];
//             }
//         }
//         if (error_status.return_code != SUCCESS)
//         {
//             goto err_solve_flux;
//         }

//         error_status = WRAP_TRACEBACK(convert_conserved_to_primitive(system));
//         if (error_status.return_code != SUCCESS)
//         {
//             goto err_convert_conserved_to_primitive;
//         }

//         error_status = WRAP_TRACEBACK(set_boundary_condition(system));
//         if (error_status.return_code != SUCCESS)
//         {
//             goto err_set_boundary_condition;
//         }

//         (*t_ptr) += dt;
//         (*num_steps_ptr)++;

//         /* Store snapshot */
//         if (is_storing && ((*t_ptr) >= storing_interval * (*store_count_ptr - store_initial_offset + 1)))
//         {
//             error_status = WRAP_TRACEBACK(store_snapshot(system, integrator_param, simulation_status, storing_param));
//             if (error_status.return_code != SUCCESS)
//             {
//                 goto err_store_snapshot;
//             }
//         }

//         if (!no_progress_bar)
//         {
//             update_progress_bar(&progress_bar_param, *t_ptr, false);
//         }
//     }

//     if (!no_progress_bar)
//     {
//         update_progress_bar(&progress_bar_param, *t_ptr, true);
//     }

//     free(temp_density_y);
//     free(temp_velocity_y_x);
//     free(temp_velocity_y_y);
//     free(temp_pressure_y);

//     free(interface_density_x_L);
//     free(interface_density_x_R);
//     free(interface_velocity_x_x_L);
//     free(interface_velocity_x_x_R);
//     free(interface_velocity_x_y_L);
//     free(interface_velocity_x_y_R);
//     free(interface_pressure_x_L);
//     free(interface_pressure_x_R);

//     free(interface_density_y_B);
//     free(interface_density_y_T);
//     free(interface_velocity_y_x_B);
//     free(interface_velocity_y_x_T);
//     free(interface_velocity_y_y_B);
//     free(interface_velocity_y_y_T);
//     free(interface_pressure_y_B);
//     free(interface_pressure_y_T);

//     return make_success_error_status();

// err_store_snapshot:
// err_set_boundary_condition:
// err_convert_conserved_to_primitive:
// err_solve_flux:
// err_dt_zero:
//     if (!no_progress_bar)
//     {
//         update_progress_bar(&progress_bar_param, *t_ptr, true);
//     }
// err_store_first_snapshot:
// err_allocate_interface_arrays:
//     free(interface_density_x_L);
//     free(interface_density_x_R);
//     free(interface_velocity_x_x_L);
//     free(interface_velocity_x_x_R);
//     free(interface_velocity_x_y_L);
//     free(interface_velocity_x_y_R);
//     free(interface_pressure_x_L);
//     free(interface_pressure_x_R);

//     free(interface_density_y_B);
//     free(interface_density_y_T);
//     free(interface_velocity_y_x_B);
//     free(interface_velocity_y_x_T);
//     free(interface_velocity_y_y_B);
//     free(interface_velocity_y_y_T);
//     free(interface_pressure_y_B);
//     free(interface_pressure_y_T);
// err_allocate_temp_arrays:
//     free(temp_density_y);
//     free(temp_velocity_y_x);
//     free(temp_velocity_y_y);
//     free(temp_pressure_y);

//     return error_status;
// }
