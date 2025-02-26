import ctypes

import numpy as np

from . import source_term
from .system import System


def solving_step(
    c_lib: ctypes.CDLL, system: System, dt: float, tol: float, solver: str
) -> None:
    """Advance the system by one time step using Godunov's first-order scheme.

    Parameters
    ----------
    c_lib : ctypes.CDLL
        C dynamic-link library object
    system : System
        System object.
    dt : float
        Time step.
    tol : float
        Tolerance for the riemann solver.
    solver : str
        Riemann solver to use, either "exact" or "hllc".

    Notes
    -----
    It is assumed that the system has been initialized with ghost cells and
    the boundary conditions have been set.
    """
    flux_mass = np.zeros(system.total_num_cells - 1)
    flux_momentum = np.zeros(system.total_num_cells - 1)
    flux_energy = np.zeros(system.total_num_cells - 1)
    c_lib.solve_system_flux(
        flux_mass.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        flux_momentum.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        flux_energy.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_double(system.gamma),
        system.density.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        system.velocity.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        system.pressure.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_double(tol),
        ctypes.c_int(system.total_num_cells),
        solver.encode("utf-8"),
    )

    # Flux exchange
    dr = system.cell_right - system.cell_left
    d_rho = flux_mass / dr[:-1] * dt
    d_rho_u = flux_momentum / dr[:-1] * dt
    d_energy_density = flux_energy / dr[:-1] * dt

    system.mass[1:-1] -= d_rho[1:] * system.volume[1:-1]
    system.momentum[1:-1] -= d_rho_u[1:] * system.volume[1:-1]
    system.energy[1:-1] -= d_energy_density[1:] * system.volume[1:-1]

    system.mass[1:-1] += d_rho[:-1] * system.volume[1:-1]
    system.momentum[1:-1] += d_rho_u[:-1] * system.volume[1:-1]
    system.energy[1:-1] += d_energy_density[:-1] * system.volume[1:-1]

    system.convert_conserved_to_primitive()
    system.set_boundary_condition()

    ### Add cylindrical / spherical geometry source term ###
    if system.coord_sys != "cartesian_1d":
        source_term.add_geometry_source_term(system, dt)
