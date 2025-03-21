cmake_minimum_required(VERSION 3.10)
project(rayleigh_taylor_2d C)

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3 -Wall -Wextra -Wpedantic")

# Debugging flags
option (DEBUG "Debug mode" OFF)
if (DEBUG)
    set(CMAKE_C_FLAGS "-O0 -g -Wall -Wextra -Wpedantic -fsanitize=address -fsanitize=undefined")
endif()

# Define source and library directories
set(SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../src)
include_directories(${SRC_DIR})

# Define the executable
add_executable(rayleigh_taylor_2d rayleigh_taylor_2d.c)

# Link against the hydro library
add_subdirectory(${SRC_DIR} build)
target_link_libraries(rayleigh_taylor_2d hydro)
