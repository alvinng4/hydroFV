"""
Riemann solver functions

References:
    1. Toro, E. F., Riemann Solvers and Numerical Methods for Fluid Dynamics,
       3rd ed. Springer., 2009.

Author: Ching-Yin Ng
Date: 2025-2-26
"""

import math
import warnings
from typing import Tuple

import numpy as np

from .exact_riemann_solver_1d import ExactRiemannSolver1D
from .hllc_riemann_solver_1d import HLLCRiemannSolver1D


def solve_system_flux(
    gamma: float,
    rho: np.ndarray,
    u: np.ndarray,
    p: np.ndarray,
    dim: int,
    solver: str,
    speed: float = 0.0,
    tol: float = 1e-6,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Check input
    available_solvers = ["exact", "hllc"]
    available_dimensions = [1]
    if solver not in available_solvers:
        raise ValueError(
            f"Invalid solver: {solver}. Available solvers: {available_solvers}"
        )
    if dim not in available_dimensions:
        raise ValueError(
            f"Invalid dimension: {dim}. Available dimensions: {available_dimensions}"
        )

    if solver == "exact":
        flux_mass, flux_momentum, flux_energy = ExactRiemannSolver1D.solve_system_flux(
            gamma, rho, u, p, speed, tol
        )
    elif solver == "hllc":
        if speed != 0.0:
            warnings.warn(
                'The "speed" parameter is available for the exact riemann solver only. Ignoring the input value.'
            )
        flux_mass, flux_momentum, flux_energy = HLLCRiemannSolver1D.solve_system_flux(
            gamma, rho, u, p, tol
        )

    return flux_mass, flux_momentum, flux_energy


def solve(
    gamma: float,
    rho_L: float,
    u_L: float,
    p_L: float,
    rho_R: float,
    u_R: float,
    p_R: float,
    dim: int,
    solver: str,
    speed: float = 0.0,
    tol: float = 1e-6,
) -> Tuple[float, float, float]:
    """Solve the Riemann problem.

    Parameters
    ----------
    gamma : float
        Adiabatic index.
    rho_L : float
        Density of the left state.
    u_L : float
        Velocity of the left state.
    p_L : float
        Pressure of the left state.
    rho_R : float
        Density of the right state.
    u_R : float
        Velocity of the right state.
    p_R : float
        Pressure of the right state.
    dim : int
        Dimension of the problem.
    solver : str
        Riemann solver to use, available options: ["exact"].
    speed : float (optional)
        Speed S = x / t when sampling at (x, t), default is 0.0.
    tol : float (optional)
        Tolerance for the Newton-Raphson method, default is 1e-6.

    Returns
    -------
    float
        Density in the middle state.
    float
        Pressure in the middle state.
    float
        Velocity in the middle state.
    """
    # Check input
    available_solvers = ["exact"]
    available_dimensions = [1]
    if solver not in available_solvers:
        if solver == "hllc":
            raise ValueError(
                f'Invalid solver: "{solver}". Use solve_system_flux() for the HLLC solver.'
            )
        else:
            raise ValueError(
                f"Invalid solver: {solver}. Available solvers: {available_solvers}"
            )
    if dim not in available_dimensions:
        raise ValueError(
            f"Invalid dimension: {dim}. Available dimensions: {available_dimensions}"
        )

    if solver == "exact":
        rho, u, p = ExactRiemannSolver1D.solve(
            gamma, rho_L, u_L, p_L, rho_R, u_R, p_R, speed, tol
        )

    return rho, u, p


def riemann_A_L_or_R(gamma: float, rho_X: float) -> float:
    """Riemann A_L or A_R function.

    Parameters
    ----------
    gamma : float
        Adiabatic index.
    rho_X : float
        Density of the left or right state.

    Returns
    -------
    float
        Value of the A_L or A_R function.
    """
    return 2.0 / ((gamma + 1.0) * rho_X)


def riemann_B_L_or_R(gamma: float, p_X: float) -> float:
    """Riemann B_L or B_R function.

    Parameters
    ----------
    gamma : float
        Adiabatic index.
    p_X : float
        Pressure of the left or right state.

    Returns
    -------
    float
        Value of the B_L or B_R function.
    """
    return ((gamma - 1.0) / (gamma + 1.0)) * p_X


def guess_p(
    gamma: float,
    rho_L: float,
    u_L: float,
    p_L: float,
    a_L: float,
    rho_R: float,
    u_R: float,
    p_R: float,
    a_R: float,
    tol: float,
) -> float:
    """Get an initial guess for the pressure in the middle state.

    Parameters
    ----------
    gamma : float
        Adiabatic index.
    rho_L : float
        Density of the left state.
    u_L : float
        Velocity of the left state.
    p_L : float
        Pressure of the left state.
    a_L : float
        Sound speed of the left state.
    rho_R : float
        Density of the right state.
    u_R : float
        Velocity of the right state.
    p_R : float
        Pressure of the right state.
    a_R : float
        Sound speed of the right state.
    tol : float
        Tolerance for the initial guess.

    Returns
    -------
    p_guess : float
        Initial guess for the pressure in the middle state.
    """
    p_min = min(p_L, p_R)
    p_max = max(p_L, p_R)

    ppv = 0.5 * (p_L + p_R) - 0.125 * (u_R - u_L) * (rho_L + rho_R) * (a_L + a_R)

    # Select PVRS Riemann solver
    if (p_max / p_min) <= 2.0 and p_min <= ppv <= p_max:
        p_guess = ppv

    # Select Two-Rarefaction Riemann solver
    elif ppv < p_min:
        gamma_minus_one = gamma - 1.0
        gamma_minus_one_over_two_gamma = gamma_minus_one / (2.0 * gamma)
        p_guess = (
            (a_L + a_R - 0.5 * gamma_minus_one * (u_R - u_L))
            / (
                a_L / (p_L**gamma_minus_one_over_two_gamma)
                + a_R / (p_R**gamma_minus_one_over_two_gamma)
            )
        ) ** (2.0 * gamma / gamma_minus_one)

    # Select Two-Shock Riemann solver with PVRS as estimate
    else:
        A_L = riemann_A_L_or_R(gamma, rho_L)
        B_L = riemann_B_L_or_R(gamma, p_L)
        g_L = math.sqrt(A_L / (ppv + B_L))

        A_R = riemann_A_L_or_R(gamma, rho_R)
        B_R = riemann_B_L_or_R(gamma, p_R)
        g_R = math.sqrt(A_R / (ppv + B_R))

        p_guess = (g_L * p_L + g_R * p_R - (u_R - u_L)) / (g_L + g_R)

    # Prevent negative value
    return max(tol, p_guess)


def solve_vacuum(
    gamma: float,
    rho_L: float,
    u_L: float,
    p_L: float,
    a_L: float,
    rho_R: float,
    u_R: float,
    p_R: float,
    a_R: float,
    speed: float = 0.0,
) -> Tuple[float, float, float]:
    """Solve the Riemann problem for vacuum.

    Parameters
    ----------
    gamma : float
        Adiabatic index.
    rho_L : float
        Density of the left state.
    u_L : float
        Velocity of the left state.
    p_L : float
        Pressure of the left state.
    rho_R : float
        Density of the right state.
    u_R : float
        Velocity of the right state.
    p_R : float
        Pressure of the right state.
    speed : float (optional)
        Speed S = x / t when sampling at (x, t), default is 0.0.

    Returns
    -------
    float
        Density in the middle state.
    float
        Pressure in the middle state.
    float
        Velocity in the middle state.
    """
    # If both states are vacuum, then the solution is also vacuum
    if rho_L <= 0.0 and rho_R <= 0.0:
        return 0.0, 0.0, 0.0

    # Right state is vacuum
    if rho_L <= 0.0:
        rho, u, p = sample_for_right_vacuum(gamma, rho_L, u_L, p_L, a_L, speed)

    # Left state is vacuum
    elif rho_R <= 0.0:
        rho, u, p = sample_for_left_vacuum(gamma, rho_L, u_L, p_L, a_L, speed)

    # Vacuum generation
    else:
        rho, u, p = sample_vacuum_generation(
            gamma, rho_L, u_L, p_L, a_L, rho_R, u_R, p_R, a_R, speed
        )

    return rho, u, p


def sample_for_right_vacuum(
    gamma: float, rho_L: float, u_L: float, p_L: float, a_L: float, speed: float
) -> Tuple[float, float, float]:
    """Sample when the right state is vacuum.

    Parameters
    ----------
    gamma : float
        Adiabatic index.
    rho_L : float
        Density of the left state.
    u_L : float
        Velocity of the left state.
    p_L : float
        Pressure of the left state.
    a_L : float
        Sound speed of the left state.
    speed : float
        Speed S = x / t when sampling at (x, t).

    Returns
    -------
    rho : float
        Density in the middle state.
    u : float
        Velocity in the middle state.
    p : float
        Pressure in the middle state.
    """
    ### Left state regime ###
    if speed <= u_L - a_L:
        rho = rho_L
        u = u_L
        p = p_L

    ### Vacuum regime ###
    else:
        # Speed of the front
        S_star_L = u_L + 2.0 * a_L / (gamma - 1.0)
        ### Rarefaction wave regime ###
        if speed < S_star_L:
            two_over_gamma_plus_one = 2.0 / (gamma + 1.0)
            common = (
                two_over_gamma_plus_one
                + ((gamma - 1.0) / ((gamma + 1.0) * a_L)) * (u_L - speed)
            ) ** (2.0 / (gamma - 1.0))

            rho = rho_L * common
            u = two_over_gamma_plus_one * (a_L + ((gamma - 1.0) / 2.0) * u_L + speed)
            p = p_L * common**gamma

        ### Vacuum regime ###
        else:
            rho = 0.0
            u = 0.0
            p = 0.0

    return rho, u, p


def sample_for_left_vacuum(
    gamma: float, rho_R: float, u_R: float, p_R: float, a_R: float, speed: float
) -> Tuple[float, float, float]:
    """Sample when the left state is vacuum.

    Parameters
    ----------
    gamma : float
        Adiabatic index.
    rho_R : float
        Density of the right state.
    u_R : float
        Velocity of the right state.
    p_R : float
        Pressure of the right state.
    a_R : float
        Sound speed of the right state.
    speed : float
        Speed S = x / t when sampling at (x, t).

    Returns
    -------
    rho : float
        Density in the middle state.
    u : float
        Velocity in the middle state.
    p : float
        Pressure in the middle state.
    """
    ### Right state regime ###
    if speed >= u_R + a_R:
        rho = rho_R
        u = u_R
        p = p_R

    ### Vacuum regime ###
    else:
        # Speed of the front
        S_star_R = u_R - 2.0 * a_R / (gamma - 1.0)

        ### Rarefaction wave regime ###
        if speed > S_star_R:
            two_over_gamma_plus_one = 2.0 / (gamma + 1.0)
            common = (
                two_over_gamma_plus_one
                - ((gamma - 1.0) / ((gamma + 1.0) * a_R)) * (u_R - speed)
            ) ** (2.0 / (gamma - 1.0))

            rho = rho_R * common
            u = two_over_gamma_plus_one * (-a_R + ((gamma - 1.0) / 2.0) * u_R + speed)
            p = p_R * common**gamma

        ### Vacuum regime ###
        else:
            rho = 0.0
            u = 0.0
            p = 0.0

    return rho, u, p


def sample_vacuum_generation(
    gamma: float,
    rho_L: float,
    u_L: float,
    p_L: float,
    a_L: float,
    rho_R: float,
    u_R: float,
    p_R: float,
    a_R: float,
    speed: float,
) -> Tuple[float, float, float]:
    """Sample for vacuum generation.

    Parameters
    ----------
    gamma : float
        Adiabatic index.
    rho_L : float
        Density of the left state.
    u_L : float
        Velocity of the left state.
    p_L : float
        Pressure of the left state.
    a_L : float
        Sound speed of the left state.
    rho_R : float
        Density of the right state.
    u_R : float
        Velocity of the right state.
    p_R : float
        Pressure of the right state.
    a_R : float
        Sound speed of the right state.
    speed : float
        Speed S = x / t when sampling at (x, t).

    Returns
    -------
    rho : float
        Density in the middle state.
    u : float
        Velocity in the middle state.
    p : float
        Pressure in the middle state.
    """
    # Speed of the left and right rarefaction waves
    S_star_L = 2.0 * a_L / (gamma - 1.0)
    S_star_R = 2.0 * a_R / (gamma - 1.0)

    # Left state
    if speed <= S_star_L:
        rho, u, p = sample_for_right_vacuum(gamma, rho_L, u_L, p_L, a_L, speed)

    elif speed <= S_star_R:
        rho = 0.0
        u = 0.0
        p = 0.0

    else:
        rho, u, p = sample_for_left_vacuum(gamma, rho_R, u_R, p_R, a_R, speed)

    return rho, u, p
