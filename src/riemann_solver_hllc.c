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
IN_FILE double compute_q_L_or_R(double gamma, double p_X, double p_star)
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

ErrorStatus solve_flux_hllc_1d(
    double *__restrict flux_mass,
    double *__restrict flux_momentum,
    double *__restrict flux_energy,
    const double gamma,
    const double rho_L,
    const double u_L,
    const double p_L,
    const double rho_R,
    const double u_R,
    const double p_R,
    const double tol
)
{
    /* Compute the sound speeds */
    double a_L;
    double a_R;
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
        rho_L <= 0.0
        || rho_R <= 0.0
        || ((a_L + a_R) * 2.0 / (gamma + 1.0)) <= (u_R - u_L)
    )
    {
        double sol_rho;
        double sol_u;
        double sol_p;
        solve_vacuum_1d(
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
        double sol_energy_density = sol_rho * (
            0.5 * sol_u * sol_u + sol_p / (sol_rho * (gamma - 1.0))
        );
        *flux_mass = sol_rho * sol_u;
        *flux_momentum = sol_rho * sol_u * sol_u + sol_p;
        *flux_energy = sol_u * (sol_energy_density + sol_p);
        return make_success_error_status();
    }

    /* Estimate the wave speeds */
    double p_star = guess_p(
        gamma, rho_L, u_L, p_L, a_L, rho_R, u_R, p_R, a_R, tol
    );

    double S_L = u_L - a_L * compute_q_L_or_R(gamma, p_L, p_star);
    double S_R = u_R + a_R * compute_q_L_or_R(gamma, p_R, p_star);

    double S_star = (p_R - p_L + rho_L * u_L * (S_L - u_L) - rho_R * u_R * (S_R - u_R)) / (
        rho_L * (S_L - u_L) - rho_R * (S_R - u_R)
    );

    /* Compute the fluxes */
    if (0.0 <= S_L)
    {
        double energy_density_L = rho_L * (
            0.5 * u_L * u_L + p_L / (rho_L * (gamma - 1.0))
        );
        *flux_mass = rho_L * u_L;
        *flux_momentum = rho_L * u_L * u_L + p_L;
        *flux_energy = u_L * (energy_density_L + p_L);
        return make_success_error_status();
    }
    else if (S_L <= 0.0 && 0.0 <= S_star)
    {
        double energy_density_L = rho_L * (
            0.5 * u_L * u_L + p_L / (rho_L * (gamma - 1.0))
        );
        double flux_mass_L = rho_L * u_L;
        double flux_momentum_L = flux_mass_L * u_L + p_L;
        double flux_energy_L = u_L * (energy_density_L + p_L);

        double rho_star_L = rho_L * (S_L - u_L) / (S_L - S_star);
        double momentum_star_L = rho_star_L * S_star;
        double energy_star_L = rho_star_L * (
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
        double energy_density_R = rho_R * (
            0.5 * u_R * u_R + p_R / (rho_R * (gamma - 1.0))
        );
        double flux_mass_R = rho_R * u_R;
        double flux_momentum_R = flux_mass_R * u_R + p_R;
        double flux_energy_R = u_R * (energy_density_R + p_R);

        double rho_star_R = rho_R * (S_R - u_R) / (S_R - S_star);
        double momentum_star_R = rho_star_R * S_star;
        double energy_star_R = rho_star_R * (
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
        double energy_density_R = rho_R * (
            0.5 * u_R * u_R + p_R / (rho_R * (gamma - 1.0))
        );
        *flux_mass = rho_R * u_R;
        *flux_momentum = rho_R * u_R * u_R + p_R;
        *flux_energy = u_R * (energy_density_R + p_R);
        return make_success_error_status();
    }
    else
    {
        return WRAP_RAISE_ERROR(VALUE_ERROR, "Invalid wave speed for the HLLC riemann solver. Possible fix: use a smaller CFL.");
    }
}

ErrorStatus solve_flux_hllc_2d(
    double *__restrict flux_mass,
    double *__restrict flux_momentum_x,
    double *__restrict flux_momentum_y,
    double *__restrict flux_energy,
    const double gamma,
    const double rho_L,
    const double u_L,
    const double v_L,
    const double p_L,
    const double rho_R,
    const double u_R,
    const double v_R,
    const double p_R,
    const double tol
)
{
    /* Compute the sound speeds */
    double a_L;
    double a_R;
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
        rho_L <= 0.0
        || rho_R <= 0.0
        || ((a_L + a_R) * 2.0 / (gamma + 1.0)) <= (u_R - u_L)
    )
    {
        double sol_rho;
        double sol_u;
        double sol_v;
        double sol_p;
        solve_vacuum_2d(
            &sol_rho,
            &sol_u,
            &sol_v,
            &sol_p,
            gamma,
            rho_L,
            u_L,
            v_L,
            p_L,
            a_L,
            rho_R,
            u_R,
            v_R,
            p_R,
            a_R,
            0.0
        );

        /* Compute the fluxes */
        double sol_energy_density = sol_rho * (
            0.5 * (sol_u * sol_u + sol_v * sol_v) + sol_p / (sol_rho * (gamma - 1.0))
        );
        *flux_mass = sol_rho * sol_u;
        *flux_momentum_x = sol_rho * sol_u * sol_u + sol_p;
        *flux_momentum_y = sol_rho * sol_u * sol_v;
        *flux_energy = sol_u * (sol_energy_density + sol_p);
        return make_success_error_status();
    }

    /* Estimate the wave speeds */
    double p_star = guess_p(
        gamma, rho_L, u_L, p_L, a_L, rho_R, u_R, p_R, a_R, tol
    );

    double S_L = u_L - a_L * compute_q_L_or_R(gamma, p_L, p_star);
    double S_R = u_R + a_R * compute_q_L_or_R(gamma, p_R, p_star);

    double S_star = (p_R - p_L + rho_L * u_L * (S_L - u_L) - rho_R * u_R * (S_R - u_R)) / (
        rho_L * (S_L - u_L) - rho_R * (S_R - u_R)
    );

    /* Compute the fluxes */
    if (0.0 <= S_L)
    {
        double energy_density_L = rho_L * (
            0.5 * (u_L * u_L + v_L * v_L) + p_L / (rho_L * (gamma - 1.0))
        );
        *flux_mass = rho_L * u_L;
        *flux_momentum_x = rho_L * u_L * u_L + p_L;
        *flux_momentum_y = rho_L * u_L * v_L;
        *flux_energy = u_L * (energy_density_L + p_L);
        return make_success_error_status();
    }
    else if (S_L <= 0.0 && 0.0 <= S_star)
    {
        double energy_density_L = rho_L * (
            0.5 * (u_L * u_L + v_L * v_L) + p_L / (rho_L * (gamma - 1.0))
        );
        double flux_mass_L = rho_L * u_L;
        double flux_momentum_L_x = flux_mass_L * u_L + p_L;
        double flux_momentum_L_y = flux_mass_L * v_L;
        double flux_energy_L = u_L * (energy_density_L + p_L);

        double rho_star_L = rho_L * (S_L - u_L) / (S_L - S_star);
        double momentum_star_L_x = rho_star_L * S_star;
        double momentum_star_L_y = rho_star_L * v_L;
        double energy_star_L = rho_star_L * (
            energy_density_L / rho_L
            + (S_star - u_L) * (S_star + p_L / (rho_L * (S_L - u_L)))
        );

        *flux_mass = flux_mass_L + S_L * (rho_star_L - rho_L);
        *flux_momentum_x = flux_momentum_L_x + S_L * (momentum_star_L_x - rho_L * u_L);
        *flux_momentum_y = flux_momentum_L_y + S_L * (momentum_star_L_y - rho_L * v_L);
        *flux_energy = flux_energy_L + S_L * (energy_star_L - energy_density_L);
        return make_success_error_status();
    }
    else if (S_star <= 0.0 && 0.0 <= S_R)
    {
        double energy_density_R = rho_R * (
            0.5 * (u_R * u_R + v_R * v_R) + p_R / (rho_R * (gamma - 1.0))
        );
        double flux_mass_R = rho_R * u_R;
        double flux_momentum_R_x = flux_mass_R * u_R + p_R;
        double flux_momentum_R_y = flux_mass_R * v_R;
        double flux_energy_R = u_R * (energy_density_R + p_R);

        double rho_star_R = rho_R * (S_R - u_R) / (S_R - S_star);
        double momentum_star_R_x = rho_star_R * S_star;
        double momentum_star_R_y = rho_star_R * v_R;
        double energy_star_R = rho_star_R * (
            energy_density_R / rho_R
            + (S_star - u_R) * (S_star + p_R / (rho_R * (S_R - u_R)))
        );

        *flux_mass = flux_mass_R + S_R * (rho_star_R - rho_R);
        *flux_momentum_x = flux_momentum_R_x + S_R * (momentum_star_R_x - rho_R * u_R);
        *flux_momentum_y = flux_momentum_R_y + S_R * (momentum_star_R_y - rho_R * v_R);
        *flux_energy = flux_energy_R + S_R * (energy_star_R - energy_density_R);
        return make_success_error_status();
    }
    else if (S_R <= 0.0)
    {
        double energy_density_R = rho_R * (
            0.5 * (u_R * u_R + v_R * v_R) + p_R / (rho_R * (gamma - 1.0))
        );
        *flux_mass = rho_R * u_R;
        *flux_momentum_x = rho_R * u_R * u_R + p_R;
        *flux_momentum_y = rho_R * u_R * v_R;
        *flux_energy = u_R * (energy_density_R + p_R);
        return make_success_error_status();
    }
    else
    {
        return WRAP_RAISE_ERROR(VALUE_ERROR, "Invalid wave speed for the HLLC riemann solver. Possible fix: use a smaller CFL.");
    }
}
