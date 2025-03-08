#ifndef PROGRESS_BAR_H
#define PROGRESS_BAR_H

#include <time.h>


typedef struct
{
    double start;
    double time_last_five_update[5];
    double last_five_progress_percent[5];
    int at_least_four_count;
    double current;
    double total;
} ProgressBarParam;

ProgressBarParam start_progress_bar(double total);
void update_progress_bar(
    ProgressBarParam *__restrict progress_bar_param,
    double current,
    bool is_end
);


#endif
