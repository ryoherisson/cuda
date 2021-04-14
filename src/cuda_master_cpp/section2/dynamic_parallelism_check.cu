#include <stdio.h>
#include <stdlib.h>

#include "../common/common.h"
#include "../common/cuda_common.cuh"

#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

__global__ void dynamic_parallelism_check(int size, int depth) {

  printf("BlockIdx.x: %d, Depth: %d, - tid : %d \n", blockIdx.x, depth,
         threadIdx.x);

  if (size == 1) {
    return;
  }

  if (threadIdx.x == 0) {
    dynamic_parallelism_check<<<1, size / 2>>>(size / 2, depth + 1);
  }
};

int main(int argc, char **argv) {

  dynamic_parallelism_check<<<2, 8>>>(8, 0);

  cudaDeviceSynchronize();
  cudaDeviceReset();
  return 0;
}