import warnings
from typing import Tuple

import numpy as np

from .exact_riemann_solver_1d import ExactRiemannSolverCartesian1D
from .hllc_riemann_solver_1d import HLLCRiemannSolverCartesian1D


def solve_system_flux(
    gamma: float,
    rho: np.ndarray,
    u: np.ndarray,
    p: np.ndarray,
    coord_sys: str,
    solver: str,
    speed: float = 0.0,
    tol: float = 1e-6,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Check input
    if coord_sys not in ["cartesian_1d", "spherical_1d"]:
        raise ValueError(
            f"Invalid coordinate system: {coord_sys}. Available coordinate systems: ['cartesian_1d', 'spherical_1d']"
        )
    if solver not in ["exact", "hllc"]:
        raise ValueError(
            f"Invalid solver: {solver}. Available solvers: ['exact', 'hllc']"
        )
    elif solver == "exact":
        if speed != 0.0:
            warnings.warn(
                'The "speed" parameter is available for the exact riemann solver only. Ignoring the input value.'
            )

    if coord_sys == "cartesian_1d":
        if solver == "exact":
            flux_mass, flux_momentum, flux_energy = (
                ExactRiemannSolverCartesian1D.solve_system_flux(
                    gamma, rho, u, p, speed, tol
                )
            )
        elif solver == "hllc":
            flux_mass, flux_momentum, flux_energy = (
                HLLCRiemannSolverCartesian1D.solve_system_flux(
                    gamma, rho, u, p, tol
                )
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
    coord_sys: str,
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
    coord_sys : str
        Coordinate system.
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
    available_coord_sys = ["cartesian_1d", "spherical_1d"]
    if coord_sys not in available_coord_sys:
        raise ValueError(
            f"Invalid coordinate system: {coord_sys}. Available coordinate systems: f{available_coord_sys}"
        )
    available_solvers = ["exact"]
    if solver not in available_solvers:
        if solver == "hllc":
            raise ValueError(
                f"Invalid solver: \"hllc\". Use solve_system_flux() for the HLLC solver."
            )
        raise ValueError(
            f"Invalid solver: \"{solver}\". Available solvers: {available_solvers}"
        )

    if coord_sys == "cartesian_1d":
        if solver == "exact":
            rho, u, p = ExactRiemannSolverCartesian1D.solve(
                gamma, rho_L, u_L, p_L, rho_R, u_R, p_R, speed, tol
            )
    elif coord_sys == "spherical_1d":
        raise NotImplementedError
    
    return rho, u, p
    