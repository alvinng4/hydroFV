from typing import Optional

import numpy as np


class System:
    AVAILABLE_BOUNDARY_CONDITIONS = ["reflective", "transmissive"]
    AVAILABLE_COORD_SYS = ["cartesian_1d", "cylindrical_1d", "spherical_1d"]

    def __init__(
        self,
        num_cells: int,
        gamma: float,
        coord_sys: str = "cartesian_1d",
        domain: tuple[float, float] = (0.0, 1.0),
        num_ghost_cells_side: int = 1,
        left_boundary_condition: Optional[str] = None,
        right_boundary_condition: Optional[str] = None,
    ) -> None:
        self.num_cells = num_cells
        self.num_ghost_cells_side = num_ghost_cells_side
        self.total_num_cells = self.num_cells + 2 * self.num_ghost_cells_side
        self.gamma = gamma

        if coord_sys not in self.AVAILABLE_COORD_SYS:
            raise ValueError(
                f"Invalid coordinate system: {coord_sys}. Available coordinate systems: {self.AVAILABLE_COORD_SYS}"
            )
        self.coord_sys = coord_sys

        if self.coord_sys == "cartesian_1d":
            self.alpha = 0.0
        elif self.coord_sys == "cylindrical_1d":
            self.alpha = 1.0
        elif self.coord_sys == "spherical_1d":
            self.alpha = 2.0

        if (
            left_boundary_condition is not None
            and left_boundary_condition not in self.AVAILABLE_BOUNDARY_CONDITIONS
        ):
            raise ValueError(
                f"Invalid left boundary condition: {left_boundary_condition}. Available boundary conditions: {self.AVAILABLE_BOUNDARY_CONDITIONS}"
            )
        if (
            right_boundary_condition is not None
            and right_boundary_condition not in self.AVAILABLE_BOUNDARY_CONDITIONS
        ):
            raise ValueError(
                f"Invalid right boundary condition: {right_boundary_condition}. Available boundary conditions: {self.AVAILABLE_BOUNDARY_CONDITIONS}"
            )

        self._left_boundary_condition = left_boundary_condition
        self._right_boundary_condition = right_boundary_condition

        self.cell_left = np.zeros(self.total_num_cells)
        self.cell_right = np.zeros(self.total_num_cells)
        self.mid_points = np.zeros(self.total_num_cells)
        self.surface_area = np.zeros(self.total_num_cells + 1)
        self.volume = np.zeros(self.total_num_cells)

        self.mass = np.zeros(self.total_num_cells)
        self.momentum = np.zeros(self.total_num_cells)
        self.energy = np.zeros(self.total_num_cells)

        self.density = np.zeros(self.total_num_cells)
        self.velocity = np.zeros(self.total_num_cells)
        self.pressure = np.zeros(self.total_num_cells)

        for i in range(self.total_num_cells):
            self.cell_left[i] = (i / self.total_num_cells) * (
                domain[1] - domain[0]
            ) + domain[0]
            self.cell_right[i] = (i + 1) / self.total_num_cells * (
                domain[1] - domain[0]
            ) + domain[0]

        self.compute_volume()
        self.compute_surface_area()
        self.compute_mid_points()

    def compute_volume(self):
        if self.coord_sys == "cartesian_1d":
            self.volume = self.cell_right - self.cell_left
        elif self.coord_sys == "cylindrical_1d":
            self.volume = np.pi * (self.cell_right**2 - self.cell_left**2)
        elif self.coord_sys == "spherical_1d":
            self.volume = 4.0 / 3.0 * np.pi * (self.cell_right**3 - self.cell_left**3)
        else:
            raise ValueError

    def compute_surface_area(self):
        if self.coord_sys == "cartesian_1d":
            self.surface_area.fill(1.0)
        elif self.coord_sys == "cylindrical_1d":
            self.surface_area[1:] = 2.0 * np.pi * self.cell_right
            self.surface_area[0] = 2.0 * np.pi * self.cell_left[0]
        elif self.coord_sys == "spherical_1d":
            self.surface_area[1:] = 4.0 * np.pi * self.cell_right**2
            self.surface_area[0] = 4.0 * np.pi * self.cell_left[0] ** 2
        else:
            raise ValueError

    def compute_mid_points(self):
        self.mid_points = 0.5 * (self.cell_left + self.cell_right)

    def set_boundary_condition(self):
        if self._left_boundary_condition is not None:
            if self._left_boundary_condition == "reflective":
                self._set_left_reflective_boundary_condition(
                    self.num_ghost_cells_side,
                    self.velocity,
                    self.pressure,
                    self.density,
                )
            elif self._left_boundary_condition == "transmissive":
                self._set_left_transmissive_boundary_condition(
                    self.num_ghost_cells_side,
                    self.velocity,
                    self.pressure,
                    self.density,
                )
            else:
                raise ValueError

        if self._right_boundary_condition is not None:
            if self._right_boundary_condition == "reflective":
                self._set_right_reflective_boundary_condition(
                    self.num_ghost_cells_side,
                    self.velocity,
                    self.pressure,
                    self.density,
                )
            elif self._right_boundary_condition == "transmissive":
                self._set_right_transmissive_boundary_condition(
                    self.num_ghost_cells_side,
                    self.velocity,
                    self.pressure,
                    self.density,
                )
            else:
                raise ValueError

    @staticmethod
    def _set_left_reflective_boundary_condition(
        num_ghost_cells_side: int,
        velocity: np.ndarray,
        pressure: np.ndarray,
        density: np.ndarray,
    ):
        for i in range(num_ghost_cells_side):
            density[num_ghost_cells_side - 1 - i] = density[num_ghost_cells_side + i]
            velocity[num_ghost_cells_side - 1 - i] = -velocity[num_ghost_cells_side + i]
            pressure[num_ghost_cells_side - 1 - i] = pressure[num_ghost_cells_side + i]

    @staticmethod
    def _set_right_reflective_boundary_condition(
        num_ghost_cells_side: int,
        velocity: np.ndarray,
        pressure: np.ndarray,
        density: np.ndarray,
    ):
        for i in range(num_ghost_cells_side):
            density[-num_ghost_cells_side + i] = density[-num_ghost_cells_side - 1 - i]
            velocity[-num_ghost_cells_side + i] = -velocity[
                -num_ghost_cells_side - 1 - i
            ]
            pressure[-num_ghost_cells_side + i] = pressure[
                -num_ghost_cells_side - 1 - i
            ]

    @staticmethod
    def _set_left_transmissive_boundary_condition(
        num_ghost_cells_side: int,
        velocity: np.ndarray,
        pressure: np.ndarray,
        density: np.ndarray,
    ):
        for i in range(num_ghost_cells_side):
            density[i] = density[num_ghost_cells_side]
            velocity[i] = velocity[num_ghost_cells_side]
            pressure[i] = pressure[num_ghost_cells_side]

    @staticmethod
    def _set_right_transmissive_boundary_condition(
        num_ghost_cells_side: int,
        velocity: np.ndarray,
        pressure: np.ndarray,
        density: np.ndarray,
    ):
        for i in range(num_ghost_cells_side):
            density[-1 - i] = density[-1 - num_ghost_cells_side]
            velocity[-1 - i] = velocity[-1 - num_ghost_cells_side]
            pressure[-1 - i] = pressure[-1 - num_ghost_cells_side]

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
