cmake_minimum_required(VERSION 3.10)
project(kelvin_helmholtz_2d C)

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3 -Wall -Wextra -Wpedantic")

# Define source and library directories
set(SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../src)
include_directories(${SRC_DIR})

# Define the executable
add_executable(kelvin_helmholtz_2d kelvin_helmholtz_2d.c)

# Link against the hydro library
add_subdirectory(${SRC_DIR} build)
target_link_libraries(kelvin_helmholtz_2d hydro)
