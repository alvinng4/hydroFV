from .system import System


def solving_step(system: System, dt: float, solver) -> None:
    """Advance the system by one time step using Godunov's first-order scheme.

    Parameters
    ----------
    system : System
        System object.
    dt : float
        Time step.
    solver
        Riemann solver.

    Notes
    -----
    It is assumed that the system has been initialized with ghost cells and
    the boundary conditions have been set.
    """
    density_sol, velocity_sol, pressure_sol = solver.solve_system(
        system.gamma, system.density, system.velocity, system.pressure
    )

    dr_arr = system.mid_points[1:] - system.mid_points[:-1]

    # Get the fluxes
    flux_mass = (
        density_sol * velocity_sol * (1.0 + 2.0 * dr_arr / system.mid_points[:-1])
    )
    flux_momentum = (
        density_sol
        * (velocity_sol * velocity_sol)
        * (1.0 + 2.0 * dr_arr / system.mid_points[:-1])
    )
    flux_momentum += pressure_sol
    flux_energy = (
        (
            pressure_sol * (system.gamma / (system.gamma - 1.0))
            + 0.5 * density_sol * (velocity_sol * velocity_sol)
        )
        * velocity_sol
        * (1.0 + 2.0 * dr_arr / system.mid_points[:-1])
    )

    # Flux exchange
    system.mass[:-1] -= flux_mass * system.surface_area[:-1] * dt
    system.momentum[:-1] -= flux_momentum * system.surface_area[:-1] * dt
    system.energy[:-1] -= flux_energy * system.surface_area[:-1] * dt

    system.mass[1:] += flux_mass * system.surface_area[1:] * dt
    system.momentum[1:] += flux_momentum * system.surface_area[1:] * dt
    system.energy[1:] += flux_energy * system.surface_area[1:] * dt

    system.convert_conserved_to_primitive()
    system.set_boundary_condition()
