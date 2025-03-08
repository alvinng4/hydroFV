/**
 * \file source_term.h
 * 
 * \brief Header file for the source term calculation
 * 
 * \author Ching-Yin Ng
 * \date 2025-03-08
 */

#ifndef SOURCE_TERM_H
#define SOURCE_TERM_H

#include "hydro.h"

ErrorStatus add_geometry_source_term(
    System *__restrict system,
    const real dt
);

#endif
