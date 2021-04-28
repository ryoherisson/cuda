#ifndef COMMON_H
#define COMMON_H

#include <cstring>
#include <fstream>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

enum INIT_PARAM {
  INIT_ZERO,
  INIT_RANDOM,
  INIT_ONE,
  INIT_ONE_TO_TEN,
  INIT_FOR_SPARSE_METRICS,
  INIT_0_TO_X
};

// simple initialization
void initialize(int *input, const int array_size,
                INIT_PARAM PARAM = INIT_ONE_TO_TEN, int x = 0);

void initialize(float *input, const int array_size,
                INIT_PARAM PARAM = INIT_ONE_TO_TEN);

// reduction in cpu
int reduction_cpu(int *input, const int size);

// compare results
void compare_results(int gpu_result, int cpu_result);

// compare two arrays
void compare_arrays(int *a, int *b, int size);

void compare_arrays(float *a, float *b, int size);

void sum_array_cpu(float *a, float *b, float *c, int size);

// print_time_using_host_clock
void print_time_using_host_clock(clock_t start, clock_t end);

#endif // !COMMON_H