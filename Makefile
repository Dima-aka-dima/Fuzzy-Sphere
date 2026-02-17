all: main

FLAGS = -Wall -Wextra -Ofast -march=native -Wno-maybe-uninitialized
LIBS = -lgsl -lgslcblas -lm
main: main.cpp 
	g++ $< -std=c++20 $(FLAGS) $(LIBS) -I./include/ -o $@
