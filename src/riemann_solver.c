/**
 * \file riemann_solver.c
 * 
 * \brief Riemann solver for 1D Euler equations.
 * 
 * \cite Toro, E. F., Riemann Solvers and Numerical Methods for Fluid Dynamics, 3rd ed. Springer., 2009.
 * 
 * \author Ching-Yin Ng
 * \date 2025-03-08
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "hydro.h"
#include "riemann_solver.h"
#include "riemann_solver_exact.h"
#include "riemann_solver_hllc.h"


ErrorStatus get_riemann_solver_flag(IntegratorParam *__restrict integrator_param)
{
    const char *riemann_solver = integrator_param->riemann_solver;
    if (strcmp(riemann_solver, "riemann_solver_exact") == 0)
    {
        integrator_param->riemann_solver_flag_ = RIEMANN_SOLVER_EXACT;
        return make_success_error_status();
    }
    else if (strcmp(riemann_solver, "riemann_solver_hllc") == 0)
    {
        integrator_param->riemann_solver_flag_ = RIEMANN_SOLVER_HLLC;
        return make_success_error_status();
    }
    else
    {
        return WRAP_RAISE_ERROR(VALUE_ERROR, "Riemann solver not recognized.");
    }
}

ErrorStatus solve_flux_1d(
    IntegratorParam *__restrict integrator_param,
    Settings *__restrict settings,
    double *__restrict flux_mass,
    double *__restrict flux_momentum,
    double *__restrict flux_energy,
    const double gamma,
    const double rho_L,
    const double u_L,
    const double p_L,
    const double rho_R,
    const double u_R,
    const double p_R
)
{
    switch (integrator_param->riemann_solver_flag_)
    {
        case RIEMANN_SOLVER_EXACT:
            return WRAP_TRACEBACK(solve_flux_exact_1d(
                flux_mass,
                flux_momentum,
                flux_energy,
                gamma,
                rho_L,
                u_L,
                p_L,
                rho_R,
                u_R,
                p_R,
                integrator_param->tol,
                0.0,
                settings->verbose
            ));
        case RIEMANN_SOLVER_HLLC:
            return WRAP_TRACEBACK(solve_flux_hllc_1d(
                flux_mass,
                flux_momentum,
                flux_energy,
                gamma,
                rho_L,
                u_L,
                p_L,
                rho_R,
                u_R,
                p_R,
                integrator_param->tol
            ));
        default:
            return WRAP_RAISE_ERROR(VALUE_ERROR, "Riemann solver flag not recognized.");
    }
}

ErrorStatus solve_flux_2d(
    IntegratorParam *__restrict integrator_param,
    Settings *__restrict settings,
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
    const double p_R
)
{
    (void) settings;
    switch (integrator_param->riemann_solver_flag_)
    {
        case RIEMANN_SOLVER_EXACT:
            return WRAP_RAISE_ERROR(VALUE_ERROR, "Only HLLC Riemann solver is available for cartesian 2D.");
        case RIEMANN_SOLVER_HLLC:
            return WRAP_TRACEBACK(solve_flux_hllc_2d(
                flux_mass,
                flux_momentum_x,
                flux_momentum_y,
                flux_energy,
                gamma,
                rho_L,
                u_L,
                v_L,
                p_L,
                rho_R,
                u_R,
                v_R,
                p_R,
                integrator_param->tol
            ));
        default:
            return WRAP_RAISE_ERROR(VALUE_ERROR, "Riemann solver flag not recognized.");
    }
}

double riemann_A_L_or_R(const double gamma, const double rho_X)
{
    return 2.0 / ((gamma + 1.0) * rho_X);
}

double riemann_B_L_or_R(const double gamma, const double p_X)
{
    return p_X * ((gamma - 1.0) / (gamma + 1.0));
}

double guess_p(
    const double gamma,
    const double rho_L,
    const double u_L,
    const double p_L,
    const double a_L,
    const double rho_R,
    const double u_R,
    const double p_R,
    const double a_R,
    const double tol
)
{
    double p_min;
    double p_max;

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

    double ppv = 0.5 * (p_L + p_R) - 0.125 * (u_R - u_L) * (rho_L + rho_R) * (a_L + a_R);
    double p_guess;

    /* Select PVRS Riemann solver */
    if (p_max / p_min <= 2.0 && (p_min <= ppv && ppv <= p_max))
    {
        p_guess = ppv;
    }

    /* Select Two-Rarefaction Riemann solver */
    else if (ppv < p_min)
    {
        double gamma_minus_one_over_two = 0.5 * (gamma - 1.0);
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
        double A_L = riemann_A_L_or_R(gamma, rho_L);
        double B_L = riemann_B_L_or_R(gamma, p_L);
        double g_L = sqrt(A_L / (ppv + B_L));

        double A_R = riemann_A_L_or_R(gamma, rho_R);
        double B_R = riemann_B_L_or_R(gamma, p_R);
        double g_R = sqrt(A_R / (ppv + B_R));

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
IN_FILE void sample_for_right_vacuum_1d(
    double *__restrict sol_rho,
    double *__restrict sol_u,
    double *__restrict sol_p,
    const double gamma,
    const double rho_L,
    const double u_L,
    const double p_L,
    const double a_L,
    const double speed
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
    double S_star_L = u_L + 2.0 * a_L / (gamma - 1.0);
    
    /* Rarefaction wave regime */
    if (speed < S_star_L)
    {
        double temp = (
            2.0 / (gamma + 1.0)
            + (u_L - speed) * (gamma - 1.0) / ((gamma + 1.0) * a_L)
        );
        temp = pow(temp, 2.0 / (gamma - 1.0));

        *sol_rho = rho_L * temp;
        *sol_u = (a_L + 0.5 * (gamma - 1.0) * u_L + speed) * 2.0 / (gamma + 1.0);
        *sol_p = p_L * pow(temp, gamma);
    }

    /* Vacuum regime */
    else
    {
        *sol_rho = 0.0;
        *sol_u = 0.0;
        *sol_p = 0.0;
    }

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
IN_FILE void sample_for_left_vacuum_1d(
    double *__restrict sol_rho,
    double *__restrict sol_u,
    double *__restrict sol_p,
    const double gamma,
    const double rho_R,
    const double u_R,
    const double p_R,
    const double a_R,
    const double speed
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
    double S_star_R = u_R - 2.0 * a_R / (gamma - 1.0);
    
    /* Rarefaction wave regime */
    if (speed > S_star_R)
    {
        double temp = (
            2.0 / (gamma + 1.0)
            - (u_R - speed) * (gamma - 1.0) / ((gamma + 1.0) * a_R)
        );
        temp = pow(temp, 2.0 / (gamma - 1.0));

        *sol_rho = rho_R * temp;
        *sol_u = (-a_R + 0.5 * (gamma - 1.0) * u_R + speed) * 2.0 / (gamma + 1.0);
        *sol_p = p_R * pow(temp, gamma);
    }

    /* Vacuum regime */
    else
    {
        *sol_rho = 0.0;
        *sol_u = 0.0;
        *sol_p = 0.0;
    }

    return;
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
IN_FILE void sample_vacuum_generation_1d(
    double *__restrict sol_rho,
    double *__restrict sol_u,
    double *__restrict sol_p,
    const double gamma,
    const double rho_L,
    const double u_L,
    const double p_L,
    const double a_L,
    const double rho_R,
    const double u_R,
    const double p_R,
    const double a_R,
    const double speed
)
{
    // Speed of the left and right rarefaction waves   
    double S_star_L = 2.0 * a_L / (gamma - 1.0);
    double S_star_R = 2.0 * a_R / (gamma - 1.0);

    /* Left state regime */
    if (speed <= S_star_L)
    {
        sample_for_right_vacuum_1d(sol_rho, sol_u, sol_p, gamma, rho_L, u_L, p_L, a_L, speed);
        return;
    }
    /* Right state regime */
    else if (S_star_R <= speed)
    {
        sample_for_left_vacuum_1d(sol_rho, sol_u, sol_p, gamma, rho_R, u_R, p_R, a_R, speed);
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

void solve_vacuum_1d(
    double *__restrict sol_rho,
    double *__restrict sol_u,
    double *__restrict sol_p,
    const double gamma,
    const double rho_L,
    const double u_L,
    const double p_L,
    const double a_L,
    const double rho_R,
    const double u_R,
    const double p_R,
    const double a_R,
    const double speed
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
        sample_for_right_vacuum_1d(sol_rho, sol_u, sol_p, gamma, rho_L, u_L, p_L, a_L, speed);
        return;
    }

    /* Left state is vacuum */
    else if (rho_R <= 0.0)
    {
        sample_for_left_vacuum_1d(sol_rho, sol_u, sol_p, gamma, rho_R, u_R, p_R, a_R, speed);
        return;
    }

    /* Vacuum generation */
    else
    {
        sample_vacuum_generation_1d(
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
IN_FILE void sample_for_right_vacuum_2d(
    double *__restrict sol_rho,
    double *__restrict sol_u,
    double *__restrict sol_v,
    double *__restrict sol_p,
    const double gamma,
    const double rho_L,
    const double u_L,
    const double v_L,
    const double p_L,
    const double a_L,
    const double speed
)
{
    /* Left state regime */
    if (speed <= (u_L - a_L))
    {
        *sol_rho = rho_L;
        *sol_u = u_L;
        *sol_v = v_L;
        *sol_p = p_L;
        return;
    }

    // Speed of the front
    double S_star_L = u_L + 2.0 * a_L / (gamma - 1.0);
    
    /* Rarefaction wave regime */
    if (speed < S_star_L)
    {
        double temp = (
            2.0 / (gamma + 1.0)
            + (u_L - speed) * (gamma - 1.0) / ((gamma + 1.0) * a_L)
        );
        temp = pow(temp, 2.0 / (gamma - 1.0));

        *sol_rho = rho_L * temp;
        *sol_u = (a_L + 0.5 * (gamma - 1.0) * u_L + speed) * 2.0 / (gamma + 1.0);
        *sol_v = v_L;
        *sol_p = p_L * pow(temp, gamma);
    }

    /* Vacuum regime */
    else
    {
        *sol_rho = 0.0;
        *sol_u = 0.0;
        *sol_v = 0.0;
        *sol_p = 0.0;
    }

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
IN_FILE void sample_for_left_vacuum_2d(
    double *__restrict sol_rho,
    double *__restrict sol_u,
    double *__restrict sol_v,
    double *__restrict sol_p,
    const double gamma,
    const double rho_R,
    const double u_R,
    const double v_R,
    const double p_R,
    const double a_R,
    const double speed
)
{
    /* Right state regime */
    if (speed >= (u_R + a_R))
    {
        *sol_rho = rho_R;
        *sol_u = u_R;
        *sol_v = v_R;
        *sol_p = p_R;
        return;
    }

    // Speed of the front
    double S_star_R = u_R - 2.0 * a_R / (gamma - 1.0);
    
    /* Rarefaction wave regime */
    if (speed > S_star_R)
    {
        double temp = (
            2.0 / (gamma + 1.0)
            - (u_R - speed) * (gamma - 1.0) / ((gamma + 1.0) * a_R)
        );
        temp = pow(temp, 2.0 / (gamma - 1.0));

        *sol_rho = rho_R * temp;
        *sol_u = (-a_R + 0.5 * (gamma - 1.0) * u_R + speed) * 2.0 / (gamma + 1.0);
        *sol_v = v_R;
        *sol_p = p_R * pow(temp, gamma);
    }

    /* Vacuum regime */
    else
    {
        *sol_rho = 0.0;
        *sol_u = 0.0;
        *sol_v = 0.0;
        *sol_p = 0.0;
    }

    return;
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
IN_FILE void sample_vacuum_generation_2d(
    double *__restrict sol_rho,
    double *__restrict sol_u,
    double *__restrict sol_v,
    double *__restrict sol_p,
    const double gamma,
    const double rho_L,
    const double u_L,
    const double v_L,
    const double p_L,
    const double a_L,
    const double rho_R,
    const double u_R,
    const double v_R,
    const double p_R,
    const double a_R,
    const double speed
)
{
    // Speed of the left and right rarefaction waves   
    double S_star_L = 2.0 * a_L / (gamma - 1.0);
    double S_star_R = 2.0 * a_R / (gamma - 1.0);

    /* Left state regime */
    if (speed <= S_star_L)
    {
        sample_for_right_vacuum_2d(sol_rho, sol_u, sol_v, sol_p, gamma, rho_L, u_L, v_L, p_L, a_L, speed);
        return;
    }
    /* Right state regime */
    else if (S_star_R <= speed)
    {
        sample_for_left_vacuum_2d(sol_rho, sol_u, sol_v, sol_p, gamma, rho_R, u_R, v_R, p_R, a_R, speed);
        return;
    }
    /* Vacuum regime */
    else
    {
        *sol_rho = 0.0;
        *sol_u = 0.0;
        *sol_v = 0.0;
        *sol_p = 0.0;
        return;
    }
}

void solve_vacuum_2d(
    double *__restrict sol_rho,
    double *__restrict sol_u,
    double *__restrict sol_v,
    double *__restrict sol_p,
    const double gamma,
    const double rho_L,
    const double u_L,
    const double v_L,
    const double p_L,
    const double a_L,
    const double rho_R,
    const double u_R,
    const double v_R,
    const double p_R,
    const double a_R,
    const double speed
)
{
    /* If both states are vacuum, then the solution is also vacuum */
    if (rho_L <= 0.0 && rho_R <= 0.0)
    {
        *sol_rho = 0.0;
        *sol_u = 0.0;
        *sol_v = 0.0;
        *sol_p = 0.0;
        return;
    }

    /* Right state is vacuum */
    else if (rho_L <= 0.0)
    {
        sample_for_right_vacuum_2d(sol_rho, sol_u, sol_v, sol_p, gamma, rho_L, u_L, v_L, p_L, a_L, speed);
        return;
    }

    /* Left state is vacuum */
    else if (rho_R <= 0.0)
    {
        sample_for_left_vacuum_2d(sol_rho, sol_u, sol_v, sol_p, gamma, rho_R, u_R, v_R, p_R, a_R, speed);
        return;
    }

    /* Vacuum generation */
    else
    {
        sample_vacuum_generation_2d(
            sol_rho,
            sol_u,
            sol_v,
            sol_p,
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
            speed
        );
        return;
    }
}
