/**
 * \file progress_bar.h
 * 
 * \brief Header file for progress bar related functions.
 * 
 * \author Ching-Yin Ng
 * \date 2025-03-11
 */

#ifndef PROGRESS_BAR_H
#define PROGRESS_BAR_H

#include <time.h>


typedef struct ProgressBarParam
{
    double start;
    double time_last_update;
    double diff_time_last_five_update[5];
    double last_five_progress_percent[5];
    int at_least_four_count;
    double current_progress;
    double total;
} ProgressBarParam;

/**
 * \brief Start the progress bar.
 * 
 * \param progress_bar_param Pointer to progress bar parameters
 * \param total Total value
 */
ErrorStatus start_progress_bar(
    ProgressBarParam *__restrict progress_bar_param,
    const double total
);

/**
 * \brief Update the progress bar.
 * 
 * \param progress_bar_param Pointer to progress bar parameters
 * \param current Current value
 * \param is_end Whether the progress bar is at the end
 */
void update_progress_bar(
    ProgressBarParam *__restrict progress_bar_param,
    double current,
    bool is_end
);


#endif
