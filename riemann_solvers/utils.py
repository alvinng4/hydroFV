import numpy as np


def get_sound_speed(
    gamma: float | np.ndarray, rho: float | np.ndarray, p: float | np.ndarray
) -> float | np.ndarray:
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
