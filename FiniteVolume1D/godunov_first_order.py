from typing import List

from . import utils
from .cell import Cell


def solving_step(cells: List[Cell], dt: float, gamma: float, solver) -> None:
    utils.convert_conserved_to_primitive(cells, gamma)
    for cell in cells:
        densityL = cell._density
        velocityL = cell._velocity
        pressureL = cell._pressure

        cell_right = cell._right_ngb
        if cell_right is None:
            # Reflective boundary condition
            densityR = densityL
            velocityR = -velocityL
            pressureR = pressureL
        else:
            densityR = cell_right._density
            velocityR = cell_right._velocity
            pressureR = cell_right._pressure

        density_sol, velocity_sol, pressure_sol = solver.solve(
            gamma, densityL, velocityL, pressureL, densityR, velocityR, pressureR
        )
        
        # Get the fluxes
        flux_mass = density_sol * velocity_sol
        flux_momentum = density_sol * (velocity_sol * velocity_sol) + pressure_sol
        flux_energy = (
            pressure_sol * (gamma / (gamma - 1.0))
            + 0.5 * density_sol * (velocity_sol * velocity_sol)
        ) * velocity_sol

        # Flux exchange
        A = cell._surface_area

        cell._mass -= flux_mass * A * dt
        cell._momentum -= flux_momentum * A * dt
        cell._energy -= flux_energy * A * dt
        
        if cell_right is not None:
            cell_right._mass += flux_mass * A * dt
            cell_right._momentum += flux_momentum * A * dt
            cell_right._energy += flux_energy * A * dt

    # Left boundary condition (reflective)
    densityL = cells[0]._density
    velocityL = -cells[0]._velocity
    pressureL = cells[0]._pressure
    densityR = cells[0]._density
    velocityR = cells[0]._velocity
    pressureR = cells[0]._pressure

    density_sol, velocity_sol, pressure_sol = solver.solve(
        gamma, densityL, velocityL, pressureL, densityR, velocityR, pressureR
    )

    # Get the fluxes
    flux_mass = density_sol * velocity_sol
    flux_momentum = density_sol * (velocity_sol * velocity_sol) + pressure_sol
    flux_energy = (
        pressure_sol * (gamma / (gamma - 1.0))
        + 0.5 * density_sol * (velocity_sol * velocity_sol)
    ) * velocity_sol

    # Flux exchange
    A = cells[0]._surface_area

    cells[0]._mass += flux_mass * A * dt
    cells[0]._momentum += flux_momentum * A * dt
    cells[0]._energy += flux_energy * A * dt
