# HydroFV

Finite volume hydrodynamics solver for the compressible Euler's equation.

## Features
- Coordinate systems:
    * Cartesian 1D, 2D
    * Cylindrical 1D
    * Spherical 1D

- Riemann solvers:
    * Exact
    * HLLC

- Integrators:
    * Random choice 1D
    * Godunov first order upwind scheme

- Extensions for Godunov scheme:
    * Piecewise linear and piecewise parabolic reconstruction
    * Minmod, Van Leer and Monotonized Central slope limiters
    * SSP-RK2 and SSP-RK3 time integration

## Requirements
Required:
- C compiler with C99 support
- HDF5 library for output

Optional:
- CMake
- OpenMP

## Sample usage
Check the `examples` directory for sample usage.
