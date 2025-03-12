/**
 * \file source_term.c
 * 
 * \brief Source term calculation for the hydrodynamics simulation
 * 
 * \author Ching-Yin Ng
 * \date 2025-03-11
 */

#include "hydro.h"

#ifdef USE_OPENMP
#include <omp.h>
#endif

/**
 * \brief Compute the source term for the cylindrical / spherical 1D geometry.
 * 
 * \param[out] d_mass Pointer to the mass source term
 * \param[out] d_momentum Pointer to the momentum source term
 * \param[out] d_energy Pointer to the energy source term
 * \param[in] pre_factor Prefactor
 * \param[in] gamma Adiabatic index
 * \param[in] density Density
 * \param[in] specific_momentum Specific momentum
 * \param[in] specific_energy Specific energy
 */
IN_FILE void _compute_geometry_source_term(
    double *__restrict d_mass,
    double *__restrict d_momentum,
    double *__restrict d_energy,
    const double pre_factor,
    const double gamma,
    const double density,
    const double specific_momentum,
    const double specific_energy
)
{
    const double pressure = (gamma - 1) * (specific_energy - 0.5 * specific_momentum * specific_momentum / density);
    *d_mass = pre_factor * specific_momentum;
    *d_momentum = pre_factor * specific_momentum * specific_momentum / density;
    *d_energy = pre_factor * specific_momentum * (specific_energy + pressure) / density;
}

ErrorStatus add_geometry_source_term(
    System *__restrict system,
    const double dt
)
{
    ErrorStatus error_status;

    /* Get alpha according to the coordinate system */
    double alpha;
    switch (system->coord_sys_flag_)
    {
        case COORD_SYS_CARTESIAN_1D: case COORD_SYS_CARTESIAN_2D: case COORD_SYS_CARTESIAN_3D:
            error_status = WRAP_RAISE_ERROR(
                VALUE_ERROR, "Wrong coordinate system, expected cylindrical 1D or spherical 1D."
            );
            goto error_coord_sys;
        case COORD_SYS_CYLINDRICAL_1D:
            alpha = 1.0;
            break;
        case COORD_SYS_SPHERICAL_1D:
            alpha = 2.0;
            break;
        default:
            error_status = WRAP_RAISE_ERROR(VALUE_ERROR, "Unknown coordinate system flag.");
            goto error_coord_sys;
    }

    const int num_ghost_cells_side = system->num_ghost_cells_side;
    const int num_cells_x = system->num_cells_x;
    const double gamma = system->gamma;
    const double *__restrict mid_points_x = system->mid_points_x_;
    const double *__restrict volume = system->volume_;
    const double *__restrict density = system->density_;
    double *__restrict mass = system->mass_;
    double *__restrict momentum = system->momentum_x_;
    double *__restrict energy = system->energy_;

    /* Compute the source term with RK4 */
#ifdef USE_OPENMP
    #pragma omp parallel for
#endif
    for (int i = num_ghost_cells_side; i < (num_cells_x + num_ghost_cells_side); i++)
    {
        double temp_mass, temp_momentum, temp_energy;
        double k1_mass, k1_momentum, k1_energy;
        double k2_mass, k2_momentum, k2_energy;
        double k3_mass, k3_momentum, k3_energy;
        double k4_mass, k4_momentum, k4_energy;

        const double volume_i = volume[i];
        const double mass_i = mass[i];
        const double momentum_i = momentum[i];
        const double energy_i = energy[i];
        const double density_i = density[i];
        const double specific_momentum_i = momentum_i / volume_i;
        const double specific_energy_i = energy_i / volume_i;

        const double pre_factor = -alpha * volume_i / mid_points_x[i];

        _compute_geometry_source_term(
            &k1_mass,
            &k1_momentum,
            &k1_energy,
            pre_factor,
            gamma,
            density_i,
            specific_momentum_i,
            specific_energy_i
        );

        temp_mass = mass_i + 0.5 * dt * k1_mass;
        temp_momentum = momentum_i + 0.5 * dt * k1_momentum;
        temp_energy = energy_i + 0.5 * dt * k1_energy;

        _compute_geometry_source_term(
            &k2_mass,
            &k2_momentum,
            &k2_energy,
            pre_factor,
            gamma,
            temp_mass / volume_i,
            temp_momentum / volume_i,
            temp_energy / volume_i
        );

        temp_mass = mass_i + 0.5 * dt * k2_mass;
        temp_momentum = momentum_i + 0.5 * dt * k2_momentum;
        temp_energy = energy_i + 0.5 * dt * k2_energy;

        _compute_geometry_source_term(
            &k3_mass,
            &k3_momentum,
            &k3_energy,
            pre_factor,
            gamma,
            temp_mass / volume_i,
            temp_momentum / volume_i,
            temp_energy / volume_i
        );

        temp_mass = mass_i + dt * k3_mass;
        temp_momentum = momentum_i + dt * k3_momentum;
        temp_energy = energy_i + dt * k3_energy;

        _compute_geometry_source_term(
            &k4_mass,
            &k4_momentum,
            &k4_energy,
            pre_factor,
            gamma,
            temp_mass / volume_i,
            temp_momentum / volume_i,
            temp_energy / volume_i
        );

        mass[i] += dt * (k1_mass + 2.0 * k2_mass + 2.0 * k3_mass + k4_mass) / 6.0;
        momentum[i] += dt * (k1_momentum + 2.0 * k2_momentum + 2.0 * k3_momentum + k4_momentum) / 6.0;
        energy[i] += dt * (k1_energy + 2.0 * k2_energy + 2.0 * k3_energy + k4_energy) / 6.0;
    }

    error_status = WRAP_TRACEBACK(convert_conserved_to_primitive(system));
    if (error_status.return_code != SUCCESS)
    {
        goto err_convert_conserved_to_primitive;
    }
    error_status = WRAP_TRACEBACK(set_boundary_condition(system));
    if (error_status.return_code != SUCCESS)
    {
        goto err_set_boundary_condition;
    }

    return make_success_error_status();

err_set_boundary_condition:
err_convert_conserved_to_primitive:
error_coord_sys:
    return error_status;
}

