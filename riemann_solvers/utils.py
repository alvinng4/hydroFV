import ctypes
import platform
from pathlib import Path
from typing import Optional

import numpy as np


def get_sound_speed(
    gamma: float | np.ndarray, rho: float | np.ndarray, p: float | np.ndarray
) -> float | np.ndarray:
    """Get the soundspeed corresponding to the given density and pressure.

    Parameters
    ----------
    gamma : float | np.ndarray
        Adiabatic index.
    rho : float | np.ndarray
        Density.
    p : float | np.ndarray
        Pressure.

    Returns
    -------
    float | np.ndarray
        Sound speed a = sqrt(gamma * p / rho).

    References
    ----------
    Toro, E. F., "The Riemann Problem for the Euler Equations" in
    Riemann Solvers and Numerical Methods for Fluid Dynamics,
    3rd ed. Springer., 2009, pp. 115-162.
    """
    return np.sqrt(gamma * p / rho)


def load_c_lib(c_lib_path: Optional[Path] = None) -> ctypes.CDLL:
    """Load the C dynamic-link library

    Returns
    -------
    c_lib : ctypes.CDLL
        C dynamic-link library object

    Raises
    ------
    OSError
        If the platform is not supported
    FileNotFoundError
        If the C library is not found at the path
    """
    if c_lib_path is None:
        if platform.system() == "Windows":
            c_lib_path = Path(__file__).parent.parent / "src" / "c_lib.dll"
        elif platform.system() == "Darwin":
            c_lib_path = Path(__file__).parent.parent / "src" / "c_lib.dylib"
        elif platform.system() == "Linux":
            c_lib_path = Path(__file__).parent.parent / "src" / "c_lib.so"
        else:
            raise Exception(
                f'Platform "{platform.system()}" not supported. Supported platforms'
                + ": Windows, macOS, Linux."
                + "You may bypass this error by providing the path to the C library"
            )

    if not Path(c_lib_path).exists():
        raise FileNotFoundError(f'C library not found at path: "{c_lib_path}"')

    return ctypes.cdll.LoadLibrary(str(c_lib_path))
