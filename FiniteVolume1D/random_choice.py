import ctypes

from . import source_term
from .system import System


class VanDerCorputSequenceGenerator:
    def __init__(self):
        self.k_1 = 5
        self.k_2 = 3
        self.count = 0

    def random(self) -> float:
        n = self.count + 1
        theta = 0
        k_1 = self.k_1
        k_2 = self.k_2

        base = self.k_1
        bk = 1 / base
        while n > 0:
            a_i = n % base
            A_i = (k_2 * a_i) % k_1
            theta += A_i * bk

            n //= base
            bk /= base

        self.count += 1
        return theta


def solving_step(
    c_lib: ctypes.CDLL,
    system: System,
    dt: float,
    tol: float,
    solver: str,
    rng: VanDerCorputSequenceGenerator,
) -> None:
    """Advance the system by one time step using Godunov's first-order scheme.

    Parameters
    ----------
    c_lib : ctypes.CDLL
        C dynamic-link library object
    system : System
        System object.
    dt : float
        Time step.
    tol : float
        Tolerance for the riemann solver.
    solver : str
        Riemann solver to use, only "exact" is available.

    Notes
    -----
    It is assumed that the system has been initialized with ghost cells and
    the boundary conditions have been set.
    """
    theta = rng.random()
    dx = system.cell_right - system.cell_left
    density_copy = system.density.copy()
    velocity_copy = system.velocity.copy()
    pressure_copy = system.pressure.copy()
    for i in range(1, system.total_num_cells - 1):
        if theta <= 0.5:
            rho_L = density_copy[i - 1]
            u_L = velocity_copy[i - 1]
            p_L = pressure_copy[i - 1]
            rho_R = density_copy[i]
            u_R = velocity_copy[i]
            p_R = pressure_copy[i]
            speed = theta * dx[i] / dt
        else:
            rho_L = density_copy[i]
            u_L = velocity_copy[i]
            p_L = pressure_copy[i]
            rho_R = density_copy[i + 1]
            u_R = velocity_copy[i + 1]
            p_R = pressure_copy[i + 1]
            speed = (theta - 1.0) * dx[i] / dt

        rho_sol_i = ctypes.c_double(0.0)
        u_sol_i = ctypes.c_double(0.0)
        p_sol_i = ctypes.c_double(0.0)
        c_lib.solve_exact(
            ctypes.byref(rho_sol_i),
            ctypes.byref(u_sol_i),
            ctypes.byref(p_sol_i),
            ctypes.c_double(system.gamma),
            ctypes.c_double(rho_L),
            ctypes.c_double(u_L),
            ctypes.c_double(p_L),
            ctypes.c_double(rho_R),
            ctypes.c_double(u_R),
            ctypes.c_double(p_R),
            ctypes.c_double(tol),
            ctypes.c_double(speed),
        )
        system.density[i] = rho_sol_i.value
        system.velocity[i] = u_sol_i.value
        system.pressure[i] = p_sol_i.value

    system.set_boundary_condition()
    system.convert_primitive_to_conserved()

    ### Add cylindrical / spherical geometry source term ###
    if system.coord_sys != "cartesian_1d":
        source_term.add_geometry_source_term(system, dt)
