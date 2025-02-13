import numpy as np

from .system import System
from .exact_riemann_solver import ExactRiemannSolver


def get_initial_system(num_cell: int) -> System:
    system = System(
        num_cells=num_cell, gamma=5.0 / 3.0, boundary_condition="reflective"
    )
    system.volume.fill(1.0 / num_cell)
    system.surface_area.fill(1.0)
    system.velocity.fill(0.0)
    for i in range(num_cell + system.num_ghosts_cells):
        system.mid_points[i] = ((i - 1) + 0.5) / num_cell

        if i < num_cell / 2:
            system.density[i] = 1.0
            system.pressure[i] = 1.0
        else:
            system.density[i] = 0.125
            system.pressure[i] = 0.1

    system.convert_primitive_to_conserved()
    system.set_boundary_condition()

    return system


def get_reference_sol(
    gamma: float, tf: float, solver: ExactRiemannSolver
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x_ref = np.arange(0.0, 1.0, 0.001)
    sol = np.array(
        [
            solver.solve(gamma, 1.0, 0.0, 1.0, 0.125, 0.0, 0.1, (x - 0.5) / tf)
            for x in x_ref
        ]
    )
    rho_ref = sol[:, 0]
    u_ref = sol[:, 1]
    p_ref = sol[:, 2]

    return x_ref, rho_ref, u_ref, p_ref
