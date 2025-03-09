CFLAGS = -O3 -std=c99 -Wall -Wextra -Wpedantic
LIBS = -lm -lhdf5
SRCS = error.c hydro.c integrator_godunov_first_order.c integrator_random_choice_1d.c integrator.c progress_bar.c riemann_solver_exact.c riemann_solver_hllc.c riemann_solver.c source_term.c system.c utils.c
OBJS = $(SRCS:.c=.o)
TARGET = libhydro.a

.PHONY: all clean

all:
	@echo "Compiling C library..."
	@$(MAKE) $(TARGET)

$(TARGET): $(OBJS)
	ar rcs $@ $^

%.o: %.c common.h
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	@echo "Removing everything except source files..."
    ifeq ($(OS),Windows_NT)
		del -f $(OBJS) $(TARGET)
    else
		rm -f $(OBJS) $(TARGET)
    endif