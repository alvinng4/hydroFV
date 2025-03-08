CFLAGS = -O3 -std=c99 -Wall -Wextra -Wpedantic
SRC_DIR = ../../src
INC = -I$(SRC_DIR)
LIBDIR = $(SRC_DIR)
LIBS = -L$(LIBDIR) -lhydro
SRCS = sedov_blast.c
OBJS = $(SRCS:.c=.o)
TARGET = sedov_blast

.PHONY: all clean

all:
	@echo "Compiling sod_shock..."
	@$(MAKE) $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LIBS) -Wl,-rpath,$(LIBDIR)

%.o: %.c
	$(CC) $(CFLAGS) $(INC) -c $< -o $@

clean:
	@echo "Removing everything except source files..."
    ifeq ($(OS),Windows_NT)
		del -f $(OBJS) $(TARGET)
    else
		rm -f $(OBJS) $(TARGET)
    endif