import math
from typing import List

import numpy as np

from .cell import Cell


def convert_conserved_to_primitive(cells: List[Cell], gamma: float):
    for cell in cells:
        cell._density = cell._mass / cell._volume
        cell._velocity = cell._momentum / cell._mass
        cell._pressure = (gamma - 1.0) * (
            cell._energy / cell._volume
            - 0.5 * cell._density * cell._velocity * cell._velocity
        )


def get_sound_speed(gamma: float | np.ndarray, rho: float | np.ndarray, p: float | np.ndarray) -> float | np.ndarray:
    """Get the soundspeed corresponding to the given density and pressure.

    Parameters
    ----------
    gamma : float | np.ndarray
        Adiabatic index.
    rho : float | np.ndarray
        Density.
    p : float | np.ndarray
        Pressure.

    Returns
    -------
    float | np.ndarray
        Sound speed a = sqrt(gamma * p / rho).

    References
    ----------
    Toro, E. F., "The Riemann Problem for the Euler Equations" in
    Riemann Solvers and Numerical Methods for Fluid Dynamics,
    3rd ed. Springer., 2009, pp. 115-162.
    """
    return np.sqrt(gamma * p / rho)


def get_time_step(cfl: float, cells: List[Cell], gamma: float) -> float:
    """Get the time step based on the CFL condition.

    Calculate dt = cfl * dx / S_max, where S_max = max{|u| + a}. Note
    that this can lead to an underestimate of S_max. For instance, at
    t = 0, if u = 0, then S_max = a_max, which results in dt thats
    too large. It is advised to use a much smaller cfl for the initial
    steps until the flow has developed.

    Parameters
    ----------
    cfl : float
        CFL number.
    cells : List[Cell]
        List of cells.
    gamma : float
        Adiabatic index.

    Returns
    -------
    float
        Time step.

    References
    ----------
    Toro, E. F., "The Riemann Problem for the Euler Equations" in
    Riemann Solvers and Numerical Methods for Fluid Dynamics,
    3rd ed. Springer., 2009, pp.221.
    """
    velocity_arr = np.array([cell._velocity for cell in cells])
    density_arr = np.array([cell._density for cell in cells])
    pressure_arr = np.array([cell._pressure for cell in cells])

    a_max = np.max(get_sound_speed(gamma, density_arr, pressure_arr))
    S_max = np.max(np.abs(velocity_arr) + a_max)
    dx = cells[1]._midpoint - cells[0]._midpoint

    return cfl * dx / S_max
