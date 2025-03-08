/**
 * \file riemann_solver_hllc.c
 * 
 * \brief HLLC Riemann solver for the 1D Euler equations.
 * 
 * \cite Toro, E. F., Riemann Solvers and Numerical Methods for Fluid Dynamics, 3rd ed. Springer., 2009.
 * 
 * \author Ching-Yin Ng
 * \date 2025-03-08
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "hydro.h"
#include "riemann_solver.h"
#include "utils.h"


/**
 * \brief Riemann q_L or q_R function.
 * 
 * \param gamma Adiabatic index.
 * \param p_X Pressure of the left or right state.
 * \param p_star Pressure in the star region.
 * 
 * \return Value of the q_L or q_R function.
 */
IN_FILE real compute_q_L_or_R(real gamma, real p_X, real p_star)
{
    if (p_star <= p_X)
    {
        return 1.0;
    }
    else
    {
        return sqrt(1.0 + (0.5 + 0.5 / gamma) * (p_star / p_X - 1.0));
    }
}

ErrorStatus solve_flux_hllc(
    real *__restrict flux_mass,
    real *__restrict flux_momentum,
    real *__restrict flux_energy,
    const real gamma,
    const real rho_L,
    const real u_L,
    const real p_L,
    const real rho_R,
    const real u_R,
    const real p_R,
    const real tol
)
{
    /* Compute the sound speeds */
    real a_L;
    real a_R;
    if (rho_L > 0.0)
    {
        a_L = get_sound_speed(gamma, rho_L, p_L);
    }
    else
    {
        // Vacuum
        // Assuming isentropic-type EOS, p = p(rho), see Toro [1]
        a_L = 0.0;
    }

    if (rho_R > 0.0)
    {
        a_R = get_sound_speed(gamma, rho_R, p_R);
    }
    else
    {
        // Vacuum
        // Assuming isentropic-type EOS, p = p(rho), see Toro [1]
        a_R = 0.0;
    }

    /**
     * Check for vacuum or vacuum generation
     * (1) If left or right states is vacuum, or
     * (2) pressure positivity condition is met
     */
    if (
        rho_L <= 0.0 ||
        rho_R <= 0.0 ||
        ((a_L + a_R) * 2.0 / (gamma + 1.0)) <= (u_R - u_L)
    )
    {
        real sol_rho;
        real sol_u;
        real sol_p;
        solve_vacuum(
            &sol_rho,
            &sol_u,
            &sol_p,
            gamma,
            rho_L,
            u_L,
            p_L,
            a_L,
            rho_R,
            u_R,
            p_R,
            a_R,
            0.0
        );

        /* Compute the fluxes */
        real sol_energy_density = sol_rho * (
            0.5 * sol_u * sol_u + sol_p / (sol_rho * (gamma - 1.0))
        );
        *flux_mass = sol_rho * sol_u;
        *flux_momentum = sol_rho * sol_u * sol_u + sol_p;
        *flux_energy = sol_u * (sol_energy_density + sol_p);
        return make_success_error_status();
    }

    /* Estimate the wave speeds */
    real p_star = guess_p(
        gamma, rho_L, u_L, p_L, a_L, rho_R, u_R, p_R, a_R, tol
    );

    real S_L = u_L - a_L * compute_q_L_or_R(gamma, p_L, p_star);
    real S_R = u_R + a_R * compute_q_L_or_R(gamma, p_R, p_star);

    real S_star = (p_R - p_L + rho_L * u_L * (S_L - u_L) - rho_R * u_R * (S_R - u_R)) / (
        rho_L * (S_L - u_L) - rho_R * (S_R - u_R)
    );

    /* Compute the fluxes */
    if (0.0 <= S_L)
    {
        real energy_density_L = rho_L * (
            0.5 * u_L * u_L + p_L / (rho_L * (gamma - 1.0))
        );
        *flux_mass = rho_L * u_L;
        *flux_momentum = rho_L * u_L * u_L + p_L;
        *flux_energy = u_L * (energy_density_L + p_L);
        return make_success_error_status();
    }
    else if (S_L <= 0.0 && 0.0 <= S_star)
    {
        real energy_density_L = rho_L * (
            0.5 * u_L * u_L + p_L / (rho_L * (gamma - 1.0))
        );
        real flux_mass_L = rho_L * u_L;
        real flux_momentum_L = flux_mass_L * u_L + p_L;
        real flux_energy_L = u_L * (energy_density_L + p_L);

        real rho_star_L = rho_L * (S_L - u_L) / (S_L - S_star);
        real momentum_star_L = rho_star_L * S_star;
        real energy_star_L = rho_star_L * (
            energy_density_L / rho_L
            + (S_star - u_L) * (S_star + p_L / (rho_L * (S_L - u_L)))
        );

        *flux_mass = flux_mass_L + S_L * (rho_star_L - rho_L);
        *flux_momentum = flux_momentum_L + S_L * (momentum_star_L - rho_L * u_L);
        *flux_energy = flux_energy_L + S_L * (energy_star_L - energy_density_L);
        return make_success_error_status();
    }
    else if (S_star <= 0.0 && 0.0 <= S_R)
    {
        real energy_density_R = rho_R * (
            0.5 * u_R * u_R + p_R / (rho_R * (gamma - 1.0))
        );
        real flux_mass_R = rho_R * u_R;
        real flux_momentum_R = flux_mass_R * u_R + p_R;
        real flux_energy_R = u_R * (energy_density_R + p_R);

        real rho_star_R = rho_R * (S_R - u_R) / (S_R - S_star);
        real momentum_star_R = rho_star_R * S_star;
        real energy_star_R = rho_star_R * (
            energy_density_R / rho_R
            + (S_star - u_R) * (S_star + p_R / (rho_R * (S_R - u_R)))
        );

        *flux_mass = flux_mass_R + S_R * (rho_star_R - rho_R);
        *flux_momentum = flux_momentum_R + S_R * (momentum_star_R - rho_R * u_R);
        *flux_energy = flux_energy_R + S_R * (energy_star_R - energy_density_R);
        return make_success_error_status();
    }
    else if (S_R <= 0.0)
    {
        real energy_density_R = rho_R * (
            0.5 * u_R * u_R + p_R / (rho_R * (gamma - 1.0))
        );
        *flux_mass = rho_R * u_R;
        *flux_momentum = rho_R * u_R * u_R + p_R;
        *flux_energy = u_R * (energy_density_R + p_R);
        return make_success_error_status();
    }
    else
    {
        return WRAP_RAISE_ERROR(VALUE_ERROR, "Invalid wave speed for the HLLC riemann solver.");
    }
}
