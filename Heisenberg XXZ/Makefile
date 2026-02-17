all: main

main: main.cpp types.hpp
	g++ $< -std=c++20 -Wall -Wextra -Ofast -march=native $(DEBUG_FLAGS) -o $@

eigen: eigen.cpp
	g++ $< -Wall -Wextra -I./include/eigen-5.0.0/ -o $@

debug: DEBUG_FLAGS = -fno-inline
debug: main
