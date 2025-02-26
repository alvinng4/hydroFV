#include <math.h>

#include "hydro.h"

real get_sound_speed(real gamma, real rho, real p)
{
    return sqrt(gamma * p / rho);
}
