from typing import List

import numpy as np

from . import utils
from .cell import Cell
from .exact_riemann_solver import ExactRiemannSolver


def cells_initial_condition(num_cell: int, gamma: float) -> list:
    cells: List[Cell] = []
    for i in range(num_cell):
        cell = Cell()
        cell._midpoint = (i + 0.5) / num_cell
        cell._volume = 1.0 / num_cell

        if i < num_cell / 2:
            cell._mass = 1.0 / num_cell
            cell._energy = (1.0 / num_cell) / (gamma - 1.0)
        else:
            cell._mass = 0.125 * (1.0 / num_cell)
            cell._energy = 0.1 * (1.0 / num_cell) / (gamma - 1.0)
        cell._momentum = 0.0

        if i > 0:
            cells[-1]._right_ngb = cell

        cell._surface_area = 1.0

        cells.append(cell)

    utils.convert_conserved_to_primitive(cells, gamma)

    return cells


def get_reference_sol(
    gamma: float, tf: float, solver: ExactRiemannSolver
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x_ref = np.arange(0.0, 1.0, 0.001)
    sol = np.array(
        [solver.solve(gamma, 1.0, 0.0, 1.0, 0.125, 0.0, 0.1, (x - 0.5) / tf) for x in x_ref]
    )
    rho_ref = sol[:, 0]
    u_ref = sol[:, 1]
    p_ref = sol[:, 2]

    return x_ref, rho_ref, u_ref, p_ref
