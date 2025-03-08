/**
 * \file riemann_solver.h
 * 
 * \brief Header file for Riemann solver for 1D Euler equations.
 * 
 * \cite Toro, E. F., Riemann Solvers and Numerical Methods for Fluid Dynamics, 3rd ed. Springer., 2009.
 * 
 * \author Ching-Yin Ng
 * \date 2025-03-08
 */

#ifndef RIEMANN_SOLVER_H
#define RIEMANN_SOLVER_H

#include "hydro.h"

/* Exact Riemann solver */
#include "riemann_solver_exact.h"

/* HLLC Riemann solver */
#include "riemann_solver_hllc.h"

#define RIEMANN_SOLVER_EXACT 0
#define RIEMANN_SOLVER_HLLC 1

/**
 * \brief Return riemann solver flag based on the input string
 * 
 * \param integrator_param Pointer to the integrator parameters.
 * 
 * \retval SUCCESS If the riemann solver is recognized
 * \retval ERROR_UNKNOWN_RIEMANN_SOLVER If the riemann solver is not recognized
 */
ErrorStatus get_riemann_solver_flag(IntegratorParam *__restrict integrator_param);

/**
 * \brief Solve the Riemann problem for flux for the whole system
 * 
 * \param integrator_param Pointer to the integrator parameters.
 * \param settings Pointer to the settings.
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
 * 
 * \retval SUCCESS If success
 * \retval error_code If error occurs
 */
ErrorStatus solve_flux(
    IntegratorParam *__restrict integrator_param,
    Settings *__restrict settings,
    real *__restrict flux_mass,
    real *__restrict flux_momentum,
    real *__restrict flux_energy,
    const real gamma,
    const real rho_L,
    const real u_L,
    const real p_L,
    const real rho_R,
    const real u_R,
    const real p_R
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
