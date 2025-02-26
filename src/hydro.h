/**
 * \file hydro.h
 * 
 * \brief Header file for the hydrodynamic module.
 * 
 * \author Ching-Yin Ng
 * \date 2025-2-26
 */

#ifndef HYDRO_H
#define HYDRO_H

// For exporting functions in Windows DLL as a dynamic-link library
#ifdef WIN32DLL_EXPORTS
    #define WIN32DLL_API __declspec(dllexport)
#else
    #define WIN32DLL_API 
#endif

#define IN_FILE static

typedef double real;


#endif