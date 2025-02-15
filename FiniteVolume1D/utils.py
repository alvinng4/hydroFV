import numpy as np
from riemann_solvers.utils import get_sound_speed

from .system import System


def get_time_step(cfl: float, system: System) -> float:
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
    system : System
        System object.

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
    a_max = get_sound_speed(system.gamma, system.density, system.pressure)
    if isinstance(a_max, np.ndarray):
        a_max = a_max.max()
    S_max = np.abs(system.velocity).max() + a_max
    dx = np.abs(np.diff(system.mid_points)).max()

    return cfl * dx / S_max
