/**
 * \file riemann_solver.h
 * 
 * \brief Header file for Riemann solver for 1D Euler equations.
 * 
 * \cite Toro, E. F., Riemann Solvers and Numerical Methods for Fluid Dynamics, 3rd ed. Springer., 2009.
 * 
 * \author Ching-Yin Ng
 * \date 2025-2-26
 */

#ifndef RIEMANN_SOLVER_H
#define RIEMANN_SOLVER_H

#include "hydro.h"

/**
 * \brief Solve the Riemann problem for flux for the whole system
 * 
 * \param flux_mass Solution array to store the mass flux.
 * \param flux_momentum Solution array to store the momentum flux.
 * \param flux_energy Solution array to store the energy flux.
 * \param gamma Adiabatic index.
 * \param rho Density array.
 * \param u Velocity array.
 * \param p Pressure array.
 * \param tol Tolerance for the riemann solver.
 * \param size Number of cells.
 * \param solver Name of the riemann solver to use.
 */
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
);

/**
 * \brief Compute the riemann A_L or A_R function.
 * 
 * \param gamma Adiabatic index.
 * \param rho_X Density of the left or right state.
 * 
 * \return Value of the riemann A_L or A_R function.
 */
real riemann_A_L_or_R(const real gamma, const real rho_X);

/**
 * \brief Compute the riemann B_L or B_R function.
 * 
 * \param gamma Adiabatic index.
 * \param p_X Pressure of the left or right state.
 * 
 * \return Value of the riemann A_L or A_R function.
 */
real riemann_B_L_or_R(const real gamma, const real p_X);

/**
 * \brief Get an initial guess for the pressure in the middle state.
 * 
 * \param gamma Adiabatic index.
 * \param rho_L Density of the left state.
 * \param u_L Velocity of the left state.
 * \param p_L Pressure of the left state.
 * \param a_L Sound speed of the left state.
 * \param rho_R Density of the right state.
 * \param u_R Velocity of the right state.
 * \param p_R Pressure of the right state.
 * \param a_R Sound speed of the right state.
 * \param tol Tolerance for the pressure.
 * 
 * \return Initial guess for the pressure in the middle state.
 */
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
);

/**
 * \brief Solve the Riemann problem for vacuum regime.
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
);




#endif
