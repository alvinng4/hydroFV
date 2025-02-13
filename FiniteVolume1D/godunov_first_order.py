from typing import List

from . import utils
from .cell import Cell

import numpy as np
def solving_step(cells: List[Cell], dt: float, gamma: float, solver) -> None:
    utils.convert_conserved_to_primitive(cells, gamma)
    density = np.array([cells[0]._density] + [cell._density for cell in cells] + [cells[-1]._density])
    velocity = np.array([cells[0]._velocity] + [cell._velocity for cell in cells] + [-cells[-1]._velocity])
    pressure = np.array([cells[0]._pressure] + [cell._pressure for cell in cells] + [cells[-1]._pressure])

    density_sol, velocity_sol, pressure_sol = solver.solve_system(
        gamma, density, velocity, pressure
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

    cells[0]._mass += flux_mass[0] * A * dt
    cells[0]._momentum += flux_momentum[0] * A * dt
    cells[0]._energy += flux_energy[0] * A * dt
    for i in range(len(cells)):
        cell = cells[i]
        cell._mass -= flux_mass[i + 1] * A * dt
        cell._momentum -= flux_momentum[i + 1] * A * dt
        cell._energy -= flux_energy[i + 1] * A * dt
        
        if i < len(cells) - 1:
            cell_right = cells[i + 1]
            cell_right._mass += flux_mass[i + 1] * A * dt
            cell_right._momentum += flux_momentum[i + 1] * A * dt
            cell_right._energy += flux_energy[i + 1] * A * dt

