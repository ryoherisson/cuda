#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "../common/common.h"
#include "../common/cuda_common.cuh"

// assume grid is 1D and block is 1D then nx = size
__global__ void sum_arrays_1Dgrid_1Dblock(float *a, float *b, float *c,
                                          int nx) {}

// assume grid is 2D and block is 2D then nx * ny = size
__global__ void sum_arrays_2Dgrid_2Dblock(float *a, float *b, float *c, int nx,
                                          int ny) {}

void sum_array_cpu(float *a, float *b, float *c, int size) {
  for (int i = 0; i < size; i++) {
    c[i] = a[i] + b[i];
  }
}

void run_sum_array_1d(int argc, char **argv) {
  printf("Running 1D grid \n");
  int size = 1 << 22;
  int block_size = 128;

  int nx, ny = 0;

  if (argc > 2) {
    size = 1 << atoi(argv[2]);
  }

  if (argc > 4) {
    block_size = 1 << atoi(argv[4]);
  }

  unsigned int byte_size = size * sizeof(float);

  printf("Input size : %d \n", size);

  float *h_a, *h_b, *h_out, *h_ref;
  h_a = (float *)malloc(byte_size);
  h_b = (float *)malloc(byte_size);
  h_out = (float *)malloc(byte_size);
  h_ref = (float *)malloc(byte_size);

  if (!h_a) {
    printf("host memory allocaiton error \n");
  }

  for (size_t i = 0; i < size; i++) {
    h_a[i] = i % 10;
    h_b[i] = i % 7;
  }

  sum_array_cpu(h_a, h_b, h_out, size);

  dim3 block(block_size);
  dim3 grid((size + block.x - 1) / block.x);

  printf("Kernel is launch with grid(%d, %d, %d) and block(%d, %d, %d) \n",
         grid.x, grid.y, grid.z, block.x, block.y, block.z);

  float *d_a, *d_b, *d_c;

  gpuErrchk(cudaMalloc((void **)&d_a, byte_size));
  gpuErrchk(cudaMalloc((void **)&d_b, byte_size));
  gpuErrchk(cudaMalloc((void **)&d_c, byte_size));
  gpuErrchk(cudaMemset(d_c, 0, byte_size));

  gpuErrchk(cudaMemcpy(d_a, h_a, byte_size, cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(d_b, h_b, byte_size, cudaMemcpyHostToDevice));

  sum_arrays_1Dgrid_1Dblock<<<grid, block>>>(d_a, d_b, d_c, size);

  gpuErrchk(cudaDeviceSynchronize());
  gpuErrchk(cudaMemcpy(h_ref, d_c, byte_size, cudaMemcpyDeviceToHost));

  compare_arrays_float(h_out, h_ref, size);

  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);
  free(h_ref);
  free(h_out);
  free(h_a);
  free(h_b);
}

void run_sum_array_2d(int argc, char **argv) {}

// arguments :
// 1 - kernel (0:1D, 1:2D)
// 2 - input size (2 pow (x))
// 3 - for 2D kernel nx,
// 4 - block.x
// 5 - block.y
int main(int argc, char **argv) {
  printf("\n------------------------SUM ARRAY EXAMPLE FOR "
         "NVPROF-------------------------\n\n");

  if (argc > 1) {
    if (atoi(argv[1]) > 0) {
      run_sum_array_2d(argc, argv);
    } else {
      run_sum_array_1d(argc, argv);
    }
  } else {
    run_sum_array_1d(argc, argv);
  }

  return 0;
}