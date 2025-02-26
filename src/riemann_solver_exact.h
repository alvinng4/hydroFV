/**
 * \file riemann_solver_exact.h
 * 
 * \brief Header file for Exact Riemann solver for 1D Euler equations.
 * 
 * \cite Toro, E. F., Riemann Solvers and Numerical Methods for Fluid Dynamics, 3rd ed. Springer., 2009.
 * 
 * \author Ching-Yin Ng
 * \date 2025-2-26
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
 */
void solve_exact(
    real *restrict sol_rho,
    real *restrict sol_u,
    real *restrict sol_p,
    const real gamma,
    const real rho_L,
    const real u_L,
    const real p_L,
    const real rho_R,
    const real u_R,
    const real p_R,
    const real tol,
    const real speed
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
 */
void solve_flux_exact(
    real *restrict flux_mass,
    real *restrict flux_momentum,
    real *restrict flux_energy,
    const real gamma,
    const real rho_L,
    const real u_L,
    const real p_L,
    const real rho_R,
    const real u_R,
    const real p_R,
    const real tol,
    const real speed
);

/**
 * \brief Solve the Riemann problem for flux for the whole system using the Exact Riemann solver.
 * 
 * \param flux_mass Solution array to store the mass flux.
 * \param flux_momentum Solution array to store the momentum flux.
 * \param flux_energy Solution array to store the energy flux.
 * \param gamma Adiabatic index.
 * \param rho_L Density of the left state.
 * \param u_L Velocity of the left state.
 * \param p_L Pressure of the left state.
 * \param rho_R Density of the right state.
 * \param u_R Velocity of the right state.
 * \param p_R Pressure of the right state.
 * \param tol Tolerance for the pressure.
 * \param size Number of cells.
 */
void solve_system_flux_exact(
    real *restrict flux_mass,
    real *restrict flux_momentum,
    real *restrict flux_energy,
    const real gamma,
    const real *restrict rho,
    const real *restrict u,
    const real *restrict p,
    const real tol,
    const int size
);

#endif
