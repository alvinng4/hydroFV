#include <stdbool.h>
#include <stdio.h>
#include <time.h>

#ifdef WIN32
    #include <windows.h>
    #include <stdint.h>
#else
    #include <sys/time.h>
#endif

#include "common.h"
#include "progress_bar.h"

#define PROGRESS_BAR_LENGTH 40

#define BULLET "\u2022"
#define BAR "\u2501"

/* 8-bit color codes */
#define RESET "\033[0m"
#define DEEP_GREEN "\033[0;32m"
#define MOCHA_GREEN "\033[38;5;106m"
#define BRIGHT_RED "\033[38;5;197m"
#define GREY "\033[0;90m"
#define YELLOW "\033[0;33m"
#define CYAN "\033[0;36m"
#define MAGENTA "\033[0;35m"


#ifdef WIN32
typedef struct timeval {
    long tv_sec;
    long tv_usec;
} timeval;

int gettimeofday(struct timeval * tp, struct timezone * tzp)
{
    // Note: some broken versions only have 8 trailing zero's, the correct epoch has 9 trailing zero's
    // This magic number is the number of 100 nanosecond intervals since January 1, 1601 (UTC)
    // until 00:00:00 January 1, 1970 
    static const uint64_t EPOCH = ((uint64_t) 116444736000000000ULL);

    SYSTEMTIME  system_time;
    FILETIME    file_time;
    uint64_t    time;

    GetSystemTime( &system_time );
    SystemTimeToFileTime( &system_time, &file_time );
    time =  ((uint64_t)file_time.dwLowDateTime )      ;
    time += ((uint64_t)file_time.dwHighDateTime) << 32;

    tp->tv_sec  = (long) ((time - EPOCH) / 10000000L);
    tp->tv_usec = (long) (system_time.wMilliseconds * 1000);
    return 0;
}
#endif

/**
 * \brief Get current time as a decimal number of seconds using clock_gettime(CLOCK_MONOTONIC, )
 * 
 * \return Current time as a decimal number of seconds
 */
IN_FILE double get_current_time(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double) ts.tv_sec + (double) ts.tv_nsec / 1.0e9;
}

IN_FILE void print_progress_bar(
    ProgressBarParam *__restrict progress_bar_param,
    double percent,
    double estimated_time_remaining,
    bool is_end
)
{
    if (percent < 0.0)
    {
        percent = 0.0;
    }
    else if (percent > 1.0)
    {
        percent = 1.0;
    }

    const int num_red_bar = (int) (percent * PROGRESS_BAR_LENGTH);
    const int num_dark_bar = PROGRESS_BAR_LENGTH - num_red_bar;

    /* Elapsed time */
    const time_t time_elapsed_time_t = get_current_time() - progress_bar_param->start;

    time_t hours_elapsed = time_elapsed_time_t / 3600;
    time_t minutes_elapsed = (time_elapsed_time_t % 3600) / 60;
    time_t seconds_elapsed = time_elapsed_time_t % 60;

    if (hours_elapsed < 0)
    {
        hours_elapsed = 0;
    }
    if (minutes_elapsed < 0)
    {
        minutes_elapsed = 0;
    }
    if (seconds_elapsed < 0)
    {
        seconds_elapsed = 0;
    }

    if (hours_elapsed > 99)
    {
        hours_elapsed = 99;
        minutes_elapsed = 59;
        seconds_elapsed = 59;
    }

    /* Remaining time */
    char remaining_time_str[9];
    const time_t estimated_time_remaining_time_t = (time_t) estimated_time_remaining;
    time_t hours_remaining = estimated_time_remaining_time_t / 3600;
    time_t minutes_remaining = (estimated_time_remaining_time_t % 3600) / 60;
    time_t seconds_remaining = estimated_time_remaining_time_t % 60;

    if (hours_remaining > 99)
    {
        hours_remaining = 99;
        minutes_remaining = 59;
        seconds_remaining = 59;
    }

    if (hours_remaining < 0 || minutes_remaining < 0 || seconds_remaining < 0)
    {
        remaining_time_str[0] = '-';
        remaining_time_str[1] = '-';
        remaining_time_str[2] = ':';
        remaining_time_str[3] = '-';
        remaining_time_str[4] = '-';
        remaining_time_str[5] = ':';
        remaining_time_str[6] = '-';
        remaining_time_str[7] = '-';
        remaining_time_str[8] = '\0';
    }
    else
    {
        snprintf(
            remaining_time_str,
            9,
            "%02d:%02d:%02d",
            (int) hours_remaining,
            (int) minutes_remaining,
            (int) seconds_remaining
        );
    }

    /* Print progress bar */
    fputs("\r\033[?25l", stdout); // Start of line and hide cursor
    if (!is_end)
    {
        fputs(BRIGHT_RED, stdout);
    }
    else
    {
        fputs(MOCHA_GREEN, stdout);
    }

    for (int i = 0; i < num_red_bar; i++)
    {
        fputs(BAR, stdout);
    }
    fputs(GREY, stdout);
    for (int i = 0; i < num_dark_bar; i++)
    {
        fputs(BAR, stdout);
    }
    printf(
        "%s %3d%%%s %s %s%02d:%02d:%02d%s %s %s%s%s",
        DEEP_GREEN,
        (int) (percent * 100),
        RESET,
        BULLET,
        YELLOW,
        (int) hours_elapsed,
        (int) minutes_elapsed,
        (int) seconds_elapsed,
        RESET,
        BULLET,
        CYAN,
        remaining_time_str,
        RESET
    );

    if (is_end)
    {
        fputs("\n", stdout); // New line
        fputs("\033[?25h", stdout); // Show cursor
    }
}

WIN32DLL_API ProgressBarParam start_progress_bar(double total)
{
    ProgressBarParam progress_bar_param;

    progress_bar_param.start = get_current_time();
    progress_bar_param.current = 0.0;
    progress_bar_param.total = total;

    progress_bar_param.last_five_progress_percent[0] = 0.0;
    progress_bar_param.time_last_five_update[0] = 0.0;
    progress_bar_param.at_least_four_count = 0;

    print_progress_bar(&progress_bar_param, 0.0, 0, false);

    return progress_bar_param;
}

IN_FILE time_t least_squares_regression_remaining_time(
    ProgressBarParam *__restrict progress_bar_param,
    const double diff_now_start
)
{
    const double target_x = 1.0;
    const double x[5] = {
        progress_bar_param->last_five_progress_percent[0],
        progress_bar_param->last_five_progress_percent[1],
        progress_bar_param->last_five_progress_percent[2],
        progress_bar_param->last_five_progress_percent[3],
        progress_bar_param->last_five_progress_percent[4]
    };
    const double y[5] = {
        progress_bar_param->time_last_five_update[0],
        progress_bar_param->time_last_five_update[1],
        progress_bar_param->time_last_five_update[2],
        progress_bar_param->time_last_five_update[3],
        progress_bar_param->time_last_five_update[4]
    };

    double sum_x = 0.0;
    double sum_y = 0.0;
    double sum_x_squared = 0.0;
    double sum_xy = 0.0;

    for (int i = 0; i < 5; i++)
    {
        sum_x += x[i];
        sum_y += y[i];
        sum_x_squared += x[i] * x[i];
        sum_xy += x[i] * y[i];
    }

    const double m = (5.0 * sum_xy - sum_x * sum_y) / (5.0 * sum_x_squared - sum_x * sum_x);
    const double b = (sum_y - m * sum_x) / 5.0;

    const double estimated_time_remaining = m * target_x + b - diff_now_start;

    return (time_t) estimated_time_remaining;
}

WIN32DLL_API void update_progress_bar(
    ProgressBarParam *__restrict progress_bar_param,
    double current,
    bool is_end
)
{
    if (current > progress_bar_param->current)
    {
        progress_bar_param->current = current;    
    }

    double percent = progress_bar_param->current / progress_bar_param->total;
    if (percent > 1.0)
    {
        percent = 1.0;
    }

    const double diff_now_start = get_current_time() - progress_bar_param->start;

    if (progress_bar_param->at_least_four_count < 4)
    {
        progress_bar_param->last_five_progress_percent[progress_bar_param->at_least_four_count] = percent;
        progress_bar_param->time_last_five_update[progress_bar_param->at_least_four_count] = diff_now_start;
        (progress_bar_param->at_least_four_count)++;
    }
    else
    {
        progress_bar_param->last_five_progress_percent[0] = progress_bar_param->last_five_progress_percent[1];
        progress_bar_param->last_five_progress_percent[1] = progress_bar_param->last_five_progress_percent[2];
        progress_bar_param->last_five_progress_percent[2] = progress_bar_param->last_five_progress_percent[3];
        progress_bar_param->last_five_progress_percent[3] = progress_bar_param->last_five_progress_percent[4];
        progress_bar_param->last_five_progress_percent[4] = percent;

        progress_bar_param->time_last_five_update[0] = progress_bar_param->time_last_five_update[1];
        progress_bar_param->time_last_five_update[1] = progress_bar_param->time_last_five_update[2];
        progress_bar_param->time_last_five_update[2] = progress_bar_param->time_last_five_update[3];
        progress_bar_param->time_last_five_update[3] = progress_bar_param->time_last_five_update[4];
        progress_bar_param->time_last_five_update[4] = diff_now_start;
    }
    
    if (!is_end)
    {
        if (progress_bar_param->at_least_four_count < 4)
        {
            print_progress_bar(progress_bar_param, percent, -1.0, false);
        }
        else
        {
            time_t estimated_time_remaining = (least_squares_regression_remaining_time(progress_bar_param, diff_now_start));
            print_progress_bar(progress_bar_param, percent, estimated_time_remaining, false);
        }
    }
    else
    {
        print_progress_bar(progress_bar_param, percent, 0.0, true);
    }
}
