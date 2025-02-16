from riemann_solvers import solve_system_flux

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
        "cartesian_1d",
        solver,
    )

    # Flux exchange
    system.mass[:-1] -= flux_mass * system.surface_area[:-1] * dt
    system.momentum[:-1] -= flux_momentum * system.surface_area[:-1] * dt
    system.energy[:-1] -= flux_energy * system.surface_area[:-1] * dt

    system.mass[1:] += flux_mass * system.surface_area[:-1] * dt
    system.momentum[1:] += flux_momentum * system.surface_area[:-1] * dt
    system.energy[1:] += flux_energy * system.surface_area[:-1] * dt

    system.convert_conserved_to_primitive()
    system.set_boundary_condition()
