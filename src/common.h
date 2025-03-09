/**
 * \file common.h
 * 
 * \brief Common definitions for the hydrodynamics simulation
 * 
 * \author Ching-Yin Ng
 * \date 2025-03-08
 */

#ifndef COMMON_H
#define COMMON_H

#include <stdbool.h>
#include <stdint.h>

// For exporting functions in Windows DLL as a dynamic-link library
#ifdef WIN32DLL_EXPORTS
#define WIN32DLL_API __declspec(dllexport)
#else
#define WIN32DLL_API 
#endif


#define IN_FILE static


typedef double real;
typedef int64_t int64;

typedef struct ErrorStatus
{
    int return_code;
    char *traceback;
    int traceback_code_;
} ErrorStatus;

typedef struct IntegratorParam
{
    const char *integrator;
    const char *riemann_solver;
    real cfl;
    real cfl_initial_shrink_factor;
    int num_steps_shrink;
    real tol;
    int integrator_flag_;
    int riemann_solver_flag_;
} IntegratorParam;

typedef struct StoringParam
{
    const char *method;
    const char *path;
    int storing_freq;
    int storing_method_flag_;
} StoringParam;

typedef struct Settings
{
    int verbose;
    bool no_progress_bar;
} Settings;

typedef struct SimulationParam
{
    real tf;
} SimulationParam;

#endif
