cmake_minimum_required(VERSION 3.10)
project(hydro C)

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3 -Wall -Wextra -Wpedantic")

# Debugging flags
option (DEBUG "Debug mode" OFF)
if (DEBUG)
    set(CMAKE_C_FLAGS "-O0 -g -Wall -Wextra -Wpedantic -fsanitize=address -fsanitize=undefined")
endif()

# OpenMP (optional)
option(USE_OPENMP "Enable OpenMP" OFF)
if(USE_OPENMP)
    find_package(OpenMP)
    if(OpenMP_C_FOUND)
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
        add_definitions(-DUSE_OPENMP)
        message(STATUS "Building with OpenMP support")
    else()
        message(WARNING "OpenMP requested but not found")
    endif()
endif()


# List of source files
set(
    SRCS
    boundary.c
    error.c
    hydro.c
    hydro_time.c
    integrator_random_choice_1d.c
    integrator_godunov_first_order_1d.c
    integrator_godunov_first_order_2d.c
    integrator.c
    progress_bar.c
    reconstruction.c
    riemann_solver_exact.c
    riemann_solver_hllc.c
    riemann_solver.c
    source_term.c
    storing.c
    system.c
    utils.c
)


# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

# Create static library
add_library(hydro STATIC ${SRCS})

# Find HDF5
find_package(HDF5 REQUIRED COMPONENTS C)

# Link libraries
target_link_libraries(hydro m ${HDF5_LIBRARIES})
if(OpenMP_C_FOUND)
    target_link_libraries(hydro ${OpenMP_C_LIBRARIES})
endif()

# Install rules
install(
    TARGETS hydro
    ARCHIVE DESTINATION lib
)