from riemann_solvers import solve_system_flux

from . import source_term
from .system import System


def solving_step(system: System, dt: float, solver: str) -> None:
    """Advance the system by one time step using Godunov's first-order scheme.

    Parameters
    ----------
    system : System
        System object.
    dt : float
        Time step.
    solver : str
        Riemann solver to use, either "exact" or "hllc".

    Notes
    -----
    It is assumed that the system has been initialized with ghost cells and
    the boundary conditions have been set.
    """
    flux_mass, flux_momentum, flux_energy = solve_system_flux(
        system.gamma,
        system.density,
        system.velocity,
        system.pressure,
        1,
        solver,
    )

    # Flux exchange
    dr = system.cell_right - system.cell_left
    d_rho = flux_mass / dr[:-1] * dt
    d_rho_u = flux_momentum / dr[:-1] * dt
    d_energy_density = flux_energy / dr[:-1] * dt

    system.mass[:-1] -= d_rho * system.volume[:-1]
    system.momentum[:-1] -= d_rho_u * system.volume[:-1]
    system.energy[:-1] -= d_energy_density * system.volume[:-1]

    system.mass[1:] += d_rho * system.volume[1:]
    system.momentum[1:] += d_rho_u * system.volume[1:]
    system.energy[1:] += d_energy_density * system.volume[1:]

    system.convert_conserved_to_primitive()
    system.set_boundary_condition()

    ### Add spherical source term ###
    if system.coord_sys == "spherical_1d":
        source_term.add_spherical_source_term(system, dt)
