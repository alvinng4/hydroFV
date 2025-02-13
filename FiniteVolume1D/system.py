import numpy as np


class System:
    AVAILABLE_BOUNDARY_CONDITIONS = ["reflective"]

    def __init__(self, num_cells: int, gamma: float, boundary_condition: str):
        self.num_cells = num_cells
        self.num_ghosts_cells = 2
        self.gamma = gamma

        if boundary_condition not in self.AVAILABLE_BOUNDARY_CONDITIONS:
            raise ValueError(
                f"Invalid boundary condition: {boundary_condition}. Available boundary conditions: {self.AVAILABLE_BOUNDARY_CONDITIONS}"
            )

        self.boundary_condition = boundary_condition

        self.mid_points = np.zeros(num_cells + self.num_ghosts_cells)
        self.volume = np.zeros(num_cells + self.num_ghosts_cells)
        self.surface_area = np.zeros(num_cells + self.num_ghosts_cells)

        self.mass = np.zeros(num_cells + self.num_ghosts_cells)
        self.momentum = np.zeros(num_cells + self.num_ghosts_cells)
        self.energy = np.zeros(num_cells + self.num_ghosts_cells)

        self.density = np.zeros(num_cells + self.num_ghosts_cells)
        self.velocity = np.zeros(num_cells + self.num_ghosts_cells)
        self.pressure = np.zeros(num_cells + self.num_ghosts_cells)

    def set_boundary_condition(self):
        if self.boundary_condition == "reflective":
            self._set_reflective_boundary_condition(
                self.velocity, self.pressure, self.density
            )
        else:
            raise ValueError(
                f"Invalid boundary condition: {self.boundary_condition}. Available boundary conditions: {self.AVAILABLE_BOUNDARY_CONDITIONS}"
            )

    @staticmethod
    def _set_reflective_boundary_condition(
        velocity: np.ndarray, pressure: np.ndarray, density: np.ndarray
    ):
        velocity[0] = -velocity[1]
        pressure[0] = pressure[1]
        density[0] = density[1]

        velocity[-1] = -velocity[-2]
        pressure[-1] = pressure[-2]
        density[-1] = density[-2]

    def convert_conserved_to_primitive(self):
        self.density = self.mass / self.volume
        self.velocity = self.momentum / self.mass
        self.pressure = (self.gamma - 1.0) * (
            self.energy / self.volume
            - 0.5 * self.density * self.velocity * self.velocity
        )
