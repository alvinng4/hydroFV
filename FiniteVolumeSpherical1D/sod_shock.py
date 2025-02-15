import numpy as np
import riemann_solvers

from .system import System


def get_initial_system(num_cells: int) -> System:
    system = System(
        num_cells=num_cells,
        gamma=5.0 / 3.0,
        left_boundary_condition="reflective",
        right_boundary_condition="reflective",
    )
    system.velocity.fill(0.0)
    for i in range(num_cells + system.num_ghosts_cells):
        system.mid_points[i] = (i + 0.5) / num_cells
        system.volume[i] = (
            4.0 / 3.0 * np.pi * (((i + 1) / num_cells) ** 3 - (i / num_cells) ** 3)
        )
        system.surface_area[i] = 4.0 * np.pi * ((i / num_cells) ** 2)

        if i < num_cells / 2:
            system.density[i] = 1.0
            system.pressure[i] = 1.0
        else:
            system.density[i] = 0.125
            system.pressure[i] = 0.1

    system.convert_primitive_to_conserved()
    system.set_boundary_condition()

    return system


def get_reference_sol(
    gamma: float, tf: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x_ref = np.arange(0.0, 1.0, 0.001)
    sol = np.array(
        [
            riemann_solvers.solve(
                gamma=gamma,
                rho_L=1.0,
                u_L=0.0,
                p_L=1.0,
                rho_R=0.125,
                u_R=0.0,
                p_R=0.1,
                speed=(x - 0.5) / tf,
                coord_sys="cartesian_1d",
                solver="exact",
            )
            for x in x_ref
        ]
    )
    rho_ref = sol[:, 0]
    u_ref = sol[:, 1]
    p_ref = sol[:, 2]

    return x_ref, rho_ref, u_ref, p_ref
