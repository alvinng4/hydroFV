/**
 * \file riemann_solver_exact.h
 * 
 * \brief Header file for Exact Riemann solver for 1D Euler equations.
 * 
 * \cite Toro, E. F., Riemann Solvers and Numerical Methods for Fluid Dynamics, 3rd ed. Springer., 2009.
 * 
 * \author Ching-Yin Ng
 * \date 2025-03-08
 */

#ifndef RIEMANN_SOLVER_EXACT_H
#define RIEMANN_SOLVER_EXACT_H

#include "hydro.h"

/**
 * \brief Solve the Riemann problem using the Exact Riemann solver.
 *
 * \param sol_rho Pointer to store the density.
 * \param sol_u Pointer to store the velocity.
 * \param sol_p Pointer to store the pressure.
 * \param gamma Adiabatic index.
 * \param rho_L Density of the left state.
 * \param u_L Velocity of the left state.
 * \param p_L Pressure of the left state.
 * \param rho_R Density of the right state.
 * \param u_R Velocity of the right state.
 * \param p_R Pressure of the right state.
 * \param tol Tolerance for the pressure.
 * \param speed Speed S = x / t for sampling at (x, t).
 * \param verbose Verbosity level.
 * 
 * \retval SUCCESS if successful
 * \retval error_code if error occurs
 */
ErrorStatus solve_exact_1d(
    double *restrict sol_rho,
    double *restrict sol_u,
    double *restrict sol_p,
    const double gamma,
    const double rho_L,
    const double u_L,
    const double p_L,
    const double rho_R,
    const double u_R,
    const double p_R,
    const double tol,
    const double speed,
    const int verbose
);

/**
 * \brief Solve the Riemann problem for flux using the Exact Riemann solver.
 * 
 * \param flux_mass Pointer to store the mass flux.
 * \param flux_momentum Pointer to store the momentum flux.
 * \param flux_energy Pointer to store the energy flux.
 * \param gamma Adiabatic index.
 * \param rho_L Density of the left state.
 * \param u_L Velocity of the left state.
 * \param p_L Pressure of the left state.
 * \param rho_R Density of the right state.
 * \param u_R Velocity of the right state.
 * \param p_R Pressure of the right state.
 * \param tol Tolerance for the pressure.
 * \param speed Speed S = x / t for sampling at (x, t).
 * \param verbose Verbosity level.
 * 
 * \retval SUCCESS if successful
 * \retval error_code if error occurs
 */
ErrorStatus solve_flux_exact_1d(
    double *restrict flux_mass,
    double *restrict flux_momentum,
    double *restrict flux_energy,
    const double gamma,
    const double rho_L,
    const double u_L,
    const double p_L,
    const double rho_R,
    const double u_R,
    const double p_R,
    const double tol,
    const double speed,
    const int verbose
);

#endif
