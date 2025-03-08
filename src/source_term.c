/**
 * \file source_term.c
 * 
 * \brief Source term calculation for the hydrodynamics simulation
 * 
 * \author Ching-Yin Ng
 * \date 2025-03-08
 */

#include "hydro.h"

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
    real *__restrict d_mass,
    real *__restrict d_momentum,
    real *__restrict d_energy,
    const real pre_factor,
    const real gamma,
    const real density,
    const real specific_momentum,
    const real specific_energy
)
{
    const real pressure = (gamma - 1) * (specific_energy - 0.5 * specific_momentum * specific_momentum / density);
    *d_mass = pre_factor * specific_momentum;
    *d_momentum = pre_factor * specific_momentum * specific_momentum / density;
    *d_energy = pre_factor * specific_momentum * (specific_energy + pressure) / density;
}

ErrorStatus add_geometry_source_term(
    System *__restrict system,
    const real dt
)
{
    ErrorStatus error_status;

    /* Get alpha according to the coordinate system */
    real alpha;
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
            error_status = WRAP_RAISE_ERROR(
                VALUE_ERROR, "Unknown coordinate system flag."
            );
            goto error_coord_sys;
    }

    const int num_ghost_cells_side = system->num_ghost_cells_side;
    const int num_cells_x = system->num_cells_x;
    const real gamma = system->gamma;
    const real *__restrict mid_points_x = system->mid_points_x_;
    const real *__restrict volume = system->volume_;
    const real *__restrict density = system->density_;
    real *__restrict mass = system->mass_;
    real *__restrict momentum = system->momentum_;
    real *__restrict energy = system->energy_;

    /* Compute the source term with RK4 */
    for (int i = num_ghost_cells_side; i < (num_cells_x + num_ghost_cells_side); i++)
    {
        real temp_mass, temp_momentum, temp_energy;
        real k1_mass, k1_momentum, k1_energy;
        real k2_mass, k2_momentum, k2_energy;
        real k3_mass, k3_momentum, k3_energy;
        real k4_mass, k4_momentum, k4_energy;

        const real volume_i = volume[i];
        const real mass_i = mass[i];
        const real momentum_i = momentum[i];
        const real energy_i = energy[i];
        const real density_i = density[i];
        const real specific_momentum_i = momentum_i / volume_i;
        const real specific_energy_i = energy_i / volume_i;

        const real pre_factor = -alpha * volume_i / mid_points_x[i];

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

    convert_conserved_to_primitive(system);
    set_boundary_condition(system);

    return make_success_error_status();

error_coord_sys:
    return error_status;
}

