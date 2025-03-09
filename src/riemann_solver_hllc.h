/**
 * \file riemann_solver_hllc.h
 * 
 * \brief Header file for HLLC Riemann solver for 1D Euler equations.
 * 
 * \cite Toro, E. F., Riemann Solvers and Numerical Methods for Fluid Dynamics, 3rd ed. Springer., 2009.
 * 
 * \author Ching-Yin Ng
 * \date 2025-03-08
 */

#ifndef RIEMANN_SOLVER_HLLC_H
#define RIEMANN_SOLVER_HLLC_H

#include "hydro.h"

/**
 * \brief Solve the 1D Riemann problem for flux using the HLLC Riemann solver.
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
 */
ErrorStatus solve_flux_hllc_1d(
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
);

ErrorStatus solve_flux_hllc_2d(
    real *__restrict flux_mass,
    real *__restrict flux_momentum_x,
    real *__restrict flux_momentum_y,
    real *__restrict flux_energy,
    const real gamma,
    const real rho_L,
    const real u_L,
    const real v_L,
    const real p_L,
    const real rho_R,
    const real u_R,
    const real v_R,
    const real p_R,
    const real tol
);

#endif
