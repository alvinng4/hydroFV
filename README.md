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
    * Minmod, Van Leer and Monotonized Central (MC) slope limiters
    * SSP-RK2 and SSP-RK3 time integration

- Todos:
    * Piecewise linear and piecewise parabolic method with characteristic tracing
    * Cartesian 3D coordinate system (easy to extend from 2d to 3d but time consuming)

## Requirements
Required:
- C compiler with C99 support
- HDF5 library for output

Optional:
- CMake
- OpenMP

## Compilation
### Using CMake
Take `examples/sod_shock_1d` as an example:
```
cmake [-DUSE_OPENMP=ON] .
cmake --build .
./sod_shock_1d
```
where `USE_OPENMP` is an optional flag to enable OpenMP parallelization.

## Sample usage
Check the `examples` directory for sample usage.
