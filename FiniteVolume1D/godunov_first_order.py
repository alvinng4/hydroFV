from .system import System


def solving_step(system: System, dt: float, solver) -> None:
    density_sol, velocity_sol, pressure_sol = solver.solve_system(
        system.gamma, system.density, system.velocity, system.pressure
    )

    # Get the fluxes
    flux_mass = density_sol * velocity_sol
    flux_momentum = density_sol * (velocity_sol * velocity_sol) + pressure_sol
    flux_energy = (
        pressure_sol * (system.gamma / (system.gamma - 1.0))
        + 0.5 * density_sol * (velocity_sol * velocity_sol)
    ) * velocity_sol

    # Flux exchange
    system.mass[:-1] -= flux_mass * system.surface_area[:-1] * dt
    system.momentum[:-1] -= flux_momentum * system.surface_area[:-1] * dt
    system.energy[:-1] -= flux_energy * system.surface_area[:-1] * dt

    system.mass[1:] += flux_mass * system.surface_area[:-1] * dt
    system.momentum[1:] += flux_momentum * system.surface_area[:-1] * dt
    system.energy[1:] += flux_energy * system.surface_area[:-1] * dt

    system.convert_conserved_to_primitive()
    system.set_boundary_condition()
