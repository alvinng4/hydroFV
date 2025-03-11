/**
 * \file riemann_solver.h
 * 
 * \brief Header file for Riemann solver for 1D Euler equations.
 * 
 * \cite Toro, E. F., Riemann Solvers and Numerical Methods for Fluid Dynamics, 3rd ed. Springer., 2009.
 * 
 * \author Ching-Yin Ng
 * \date 2025-03-11
 */

#ifndef RIEMANN_SOLVER_H
#define RIEMANN_SOLVER_H

#include "hydro.h"

/* Exact Riemann solver */
#include "riemann_solver_exact.h"

/* HLLC Riemann solver */
#include "riemann_solver_hllc.h"

#define RIEMANN_SOLVER_EXACT 1
#define RIEMANN_SOLVER_HLLC 2

/**
 * \brief Set riemann solver flag based on the input string
 * 
 * \param integrator_param Pointer to the integrator parameters.
 * 
 * \return ErrorStatus struct.
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
);

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
);

/**
 * \brief Compute the riemann A_L or A_R function.
 * 
 * \param gamma Adiabatic index.
 * \param rho_X Density of the left or right state.
 * 
 * \return Value of the riemann A_L or A_R function.
 */
double riemann_A_L_or_R(const double gamma, const double rho_X);

/**
 * \brief Compute the riemann B_L or B_R function.
 * 
 * \param gamma Adiabatic index.
 * \param p_X Pressure of the left or right state.
 * 
 * \return Value of the riemann A_L or A_R function.
 */
double riemann_B_L_or_R(const double gamma, const double p_X);

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
);

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
);


#endif
