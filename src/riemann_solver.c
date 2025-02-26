/**
 * \file riemann_solver.c
 * 
 * \brief Riemann solver for 1D Euler equations.
 * 
 * \cite Toro, E. F., Riemann Solvers and Numerical Methods for Fluid Dynamics, 3rd ed. Springer., 2009.
 * 
 * \author Ching-Yin Ng
 * \date 2025-2-26
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "hydro.h"
#include "riemann_solver.h"
#include "riemann_solver_exact.h"
#include "riemann_solver_hllc.h"


void solve_system_flux(
    real *restrict flux_mass,
    real *restrict flux_momentum,
    real *restrict flux_energy,
    const real gamma,
    const real *restrict rho,
    const real *restrict u,
    const real *restrict p,
    const real tol,
    const int size,
    const char *restrict solver
)
{
    if (strcmp(solver, "exact") == 0)
    {
        solve_system_flux_exact(
            flux_mass,
            flux_momentum,
            flux_energy,
            gamma,
            rho,
            u,
            p,
            tol,
            size
        );
    }
    else if (strcmp(solver, "hllc") == 0)
    {
        solve_system_flux_hllc(
            flux_mass,
            flux_momentum,
            flux_energy,
            gamma,
            rho,
            u,
            p,
            tol,
            size
        );
    }
    else
    {
        fprintf(stderr, "Error: Riemann solver not recognized.\n");
        exit(EXIT_FAILURE);
    }
}

real riemann_A_L_or_R(const real gamma, const real rho_X)
{
    return 2.0 / ((gamma + 1.0) * rho_X);
}

real riemann_B_L_or_R(const real gamma, const real p_X)
{
    return p_X * ((gamma - 1.0) / (gamma + 1.0));
}

real guess_p(
    const real gamma,
    const real rho_L,
    const real u_L,
    const real p_L,
    const real a_L,
    const real rho_R,
    const real u_R,
    const real p_R,
    const real a_R,
    const real tol
)
{
    real p_min;
    real p_max;

    if (p_L > p_R)
    {
        p_min = p_R;
        p_max = p_L;
    }
    else
    {
        p_min = p_L;
        p_max = p_R;
    }

    real ppv = 0.5 * (p_L + p_R) - 0.125 * (u_R - u_L) * (rho_L + rho_R) * (a_L + a_R);
    real p_guess;

    /* Select PVRS Riemann solver */
    if (p_max / p_min <= 2.0 && (p_min <= ppv && ppv <= p_max))
    {
        p_guess = ppv;
    }

    /* Select Two-Rarefaction Riemann solver */
    else if (ppv < p_min)
    {
        real gamma_minus_one_over_two = 0.5 * (gamma - 1.0);
        p_guess = (
            a_L + a_R - gamma_minus_one_over_two * (u_R - u_L)
        ) / (
            a_L / (pow(p_L, gamma_minus_one_over_two / gamma))
            + a_R / (pow(p_R, gamma_minus_one_over_two / gamma))
        );
        p_guess = pow(p_guess, gamma / gamma_minus_one_over_two);
    }

    /* Select Two-Shock Riemann solver with PVRS as estimate */
    else
    {
        real A_L = riemann_A_L_or_R(gamma, rho_L);
        real B_L = riemann_B_L_or_R(gamma, p_L);
        real g_L = sqrt(A_L / (ppv + B_L));

        real A_R = riemann_A_L_or_R(gamma, rho_R);
        real B_R = riemann_B_L_or_R(gamma, p_R);
        real g_R = sqrt(A_R / (ppv + B_R));

        p_guess = (g_L * p_L + g_R * p_R - (u_R - u_L)) / (g_L + g_R);
    }

    /* Prevent negative value */
    if (p_guess < tol)
    {
        p_guess = tol;
    }

    return p_guess;
}

/**
 * \brief Sample the riemann problem solution for the right vacuum regime.
 * 
 * \param sol_rho Pointer to the density solution.
 * \param sol_u Pointer to the velocity solution.
 * \param sol_p Pointer to the pressure solution.
 * \param gamma Adiabatic index.
 * \param rho_L Density of the left state.
 * \param u_L Velocity of the left state.
 * \param p_L Pressure of the left state.
 * \param a_L Sound speed of the left state.
 * \param speed Speed S = x / t for sampling at (x, t).
 */
IN_FILE void sample_for_right_vacuum(
    real *restrict sol_rho,
    real *restrict sol_u,
    real *restrict sol_p,
    real gamma,
    real rho_L,
    real u_L,
    real p_L,
    real a_L,
    real speed
)
{
    /* Left state regime */
    if (speed <= (u_L - a_L))
    {
        *sol_rho = rho_L;
        *sol_u = u_L;
        *sol_p = p_L;
        return;
    }

    // Speed of the front
    real S_star_L = u_L + 2.0 * a_L / (gamma - 1.0);
    
    /* Rarefaction wave regime */
    if (speed < S_star_L)
    {
        real temp = (
            2.0 / (gamma + 1.0)
            + (u_L - speed) * (gamma - 1.0) / ((gamma + 1.0) * a_L)
        );
        temp = pow(temp, 2.0 / (gamma - 1.0));

        *sol_rho = rho_L * temp;
        *sol_u = (a_L + 0.5 * (gamma - 1.0) * u_L + speed) * 2.0 / (gamma + 1.0);
        *sol_p = p_L * pow(temp, gamma);
        return;
    }
    
    /* Vacuum regime */
    *sol_rho = 0.0;
    *sol_u = 0.0;
    *sol_p = 0.0;
    return;
}

/**
 * \brief Sample the riemann problem solution for the left vacuum regime.
 * 
 * \param sol_rho Pointer to the density solution.
 * \param sol_u Pointer to the velocity solution.
 * \param sol_p Pointer to the pressure solution.
 * \param gamma Adiabatic index.
 * \param rho_R Density of the right state.
 * \param u_R Velocity of the right state.
 * \param p_R Pressure of the right state.
 * \param a_R Sound speed of the right state.
 * \param speed Speed S = x / t for sampling at (x, t).
 */
IN_FILE void sample_for_left_vacuum(
    real *restrict sol_rho,
    real *restrict sol_u,
    real *restrict sol_p,
    real gamma,
    real rho_R,
    real u_R,
    real p_R,
    real a_R,
    real speed
)
{
    /* Right state regime */
    if (speed >= (u_R + a_R))
    {
        *sol_rho = rho_R;
        *sol_u = u_R;
        *sol_p = p_R;
        return;
    }

    // Speed of the front
    real S_star_R = u_R - 2.0 * a_R / (gamma - 1.0);
    
    /* Rarefaction wave regime */
    if (speed > S_star_R)
    {
        real temp = (
            2.0 / (gamma + 1.0)
            - (u_R - speed) * (gamma - 1.0) / ((gamma + 1.0) * a_R)
        );
        temp = pow(temp, 2.0 / (gamma - 1.0));

        *sol_rho = rho_R * temp;
        *sol_u = (-a_R + 0.5 * (gamma - 1.0) * u_R + speed) * 2.0 / (gamma + 1.0);
        *sol_p = p_R * pow(temp, gamma);
        return;
    }
    /* Vacuum regime */
    else
    {
        *sol_rho = 0.0;
        *sol_u = 0.0;
        *sol_p = 0.0;
        return;
    }
}

/**
 * \brief Sample the riemann problem solution for vacuum generation.
 * 
 * \param sol_rho Pointer to the density solution.
 * \param sol_u Pointer to the velocity solution.
 * \param sol_p Pointer to the pressure solution.
 * \param gamma Adiabatic index.
 * \param rho_L Density of the left state.
 * \param u_L Velocity of the left state.
 * \param p_L Pressure of the left state.
 * \param a_L Sound speed of the left state.
 * \param rho_R Density of the right state.
 * \param u_R Velocity of the right state.
 * \param p_R Pressure of the right state.
 * \param a_R Sound speed of the right state.
 * \param speed Speed S = x / t for sampling at (x, t).
 */
IN_FILE void sample_vacuum_generation(
    real *restrict sol_rho,
    real *restrict sol_u,
    real *restrict sol_p,
    real gamma,
    real rho_L,
    real u_L,
    real p_L,
    real a_L,
    real rho_R,
    real u_R,
    real p_R,
    real a_R,
    real speed
)
{
    // Speed of the left and right rarefaction waves   
    real S_star_L = 2.0 * a_L / (gamma - 1.0);
    real S_star_R = 2.0 * a_R / (gamma - 1.0);

    /* Left state regime */
    if (speed <= S_star_L)
    {
        sample_for_right_vacuum(sol_rho, sol_u, sol_p, gamma, rho_L, u_L, p_L, a_L, speed);
        return;
    }
    /* Right state regime */
    else if (S_star_R <= speed)
    {
        sample_for_left_vacuum(sol_rho, sol_u, sol_p, gamma, rho_R, u_R, p_R, a_R, speed);
        return;
    }
    /* Vacuum regime */
    else
    {
        *sol_rho = 0.0;
        *sol_u = 0.0;
        *sol_p = 0.0;
        return;
    }
}

void solve_vacuum(
    real *restrict sol_rho,
    real *restrict sol_u,
    real *restrict sol_p,
    const real gamma,
    const real rho_L,
    const real u_L,
    const real p_L,
    const real a_L,
    const real rho_R,
    const real u_R,
    const real p_R,
    const real a_R,
    const real speed
)
{
    /* If both states are vacuum, then the solution is also vacuum */
    if (rho_L <= 0.0 && rho_R <= 0.0)
    {
        *sol_rho = 0.0;
        *sol_u = 0.0;
        *sol_p = 0.0;
        return;
    }

    /* Right state is vacuum */
    else if (rho_L <= 0.0)
    {
        sample_for_right_vacuum(sol_rho, sol_u, sol_p, gamma, rho_L, u_L, p_L, a_L, speed);
        return;
    }

    /* Left state is vacuum */
    else if (rho_R <= 0.0)
    {
        sample_for_left_vacuum(sol_rho, sol_u, sol_p, gamma, rho_R, u_R, p_R, a_R, speed);
        return;
    }

    /* Vacuum generation */
    else
    {
        sample_vacuum_generation(
            sol_rho,
            sol_u,
            sol_p,
            gamma,
            rho_L,
            u_L,
            p_L,
            a_L,
            rho_R,
            u_R,
            p_R,
            a_R,
            speed
        );
        return;
    }
}
