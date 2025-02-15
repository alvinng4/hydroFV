import numpy as np


class System:
    AVAILABLE_BOUNDARY_CONDITIONS = ["reflective", "transmissive"]

    def __init__(
        self,
        num_cells: int,
        gamma: float,
        left_boundary_condition: str,
        right_boundary_condition: str,
    ):
        self.num_cells = num_cells
        self.num_ghosts_cells = 2
        self.gamma = gamma

        if left_boundary_condition not in self.AVAILABLE_BOUNDARY_CONDITIONS:
            raise ValueError(
                f"Invalid left boundary condition: {left_boundary_condition}. Available boundary conditions: {self.AVAILABLE_BOUNDARY_CONDITIONS}"
            )
        if right_boundary_condition not in self.AVAILABLE_BOUNDARY_CONDITIONS:
            raise ValueError(
                f"Invalid right boundary condition: {right_boundary_condition}. Available boundary conditions: {self.AVAILABLE_BOUNDARY_CONDITIONS}"
            )

        self._left_boundary_condition = left_boundary_condition
        self._right_boundary_condition = right_boundary_condition

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
        if self._left_boundary_condition == "reflective":
            self._set_left_reflective_boundary_condition(
                self.velocity, self.pressure, self.density
            )
        elif self._left_boundary_condition == "transmissive":
            self._set_left_transmissive_boundary_condition(
                self.velocity, self.pressure, self.density
            )
        else:
            raise ValueError

        if self._right_boundary_condition == "reflective":
            self._set_right_reflective_boundary_condition(
                self.velocity, self.pressure, self.density
            )
        elif self._right_boundary_condition == "transmissive":
            self._set_right_transmissive_boundary_condition(
                self.velocity, self.pressure, self.density
            )
        else:
            raise ValueError

    @staticmethod
    def _set_left_reflective_boundary_condition(
        velocity: np.ndarray, pressure: np.ndarray, density: np.ndarray
    ):
        velocity[0] = -velocity[1]
        pressure[0] = pressure[1]
        density[0] = density[1]

    @staticmethod
    def _set_right_reflective_boundary_condition(
        velocity: np.ndarray, pressure: np.ndarray, density: np.ndarray
    ):
        velocity[-1] = -velocity[-2]
        pressure[-1] = pressure[-2]
        density[-1] = density[-2]

    @staticmethod
    def _set_left_transmissive_boundary_condition(
        velocity: np.ndarray, pressure: np.ndarray, density: np.ndarray
    ):
        velocity[0] = velocity[1]
        pressure[0] = pressure[1]
        density[0] = density[1]

    @staticmethod
    def _set_right_transmissive_boundary_condition(
        velocity: np.ndarray, pressure: np.ndarray, density: np.ndarray
    ):
        velocity[-1] = velocity[-2]
        pressure[-1] = pressure[-2]
        density[-1] = density[-2]

    def convert_conserved_to_primitive(self):
        self.density = self.mass / self.volume
        self.velocity = self.momentum / self.mass
        self.pressure = (self.gamma - 1.0) * (
            self.energy / self.volume
            - 0.5 * self.density * self.velocity * self.velocity
        )

    def convert_primitive_to_conserved(self):
        self.mass = self.density * self.volume
        self.momentum = self.mass * self.velocity
        self.energy = self.volume * (
            0.5 * self.density * self.velocity * self.velocity
            + self.pressure / (self.gamma - 1.0)
        )
