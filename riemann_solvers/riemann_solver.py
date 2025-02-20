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
                f'Invalid solver: "hllc". Use solve_system_flux() for the HLLC solver.'
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
