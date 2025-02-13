from .system import System


def solving_step(system: System, dt: float, solver) -> None:
    system.convert_conserved_to_primitive()
    system.set_boundary_condition()
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
    # system.mass[0] += flux_mass[0] * system.surface_area[0] * dt
    # system.momentum[0] += flux_momentum[0] * system.surface_area[0] * dt
    # system.energy[0] += flux_energy[0] * system.surface_area[0] * dt
    for i in range(system.num_cells + 1):
        system.mass[i] -= flux_mass[i] * system.surface_area[i] * dt
        system.momentum[i] -= flux_momentum[i] * system.surface_area[i] * dt
        system.energy[i] -= flux_energy[i] * system.surface_area[i] * dt

        system.mass[i + 1] += flux_mass[i] * system.surface_area[i] * dt
        system.momentum[i + 1] += flux_momentum[i] * system.surface_area[i] * dt
        system.energy[i + 1] += flux_energy[i] * system.surface_area[i] * dt

    system.convert_conserved_to_primitive()
    system.set_boundary_condition()
