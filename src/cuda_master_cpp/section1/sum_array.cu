// #include "cuda_common.cuh"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <time.h>
// for memset
#include <cstring>

#include "../common/common.h"
#include "../common/cuda_common.cuh"

__global__ void sum_array_gpu(int *a, int *b, int *c, int size) {
  int gid = blockIdx.x * blockDim.x + threadIdx.x;

  if (gid < size) {
    c[gid] = a[gid] + b[gid];
  }
}

void sum_array_cpu(int *a, int *b, int *c, int size) {
  for (int i = 0; i < size; i++) {
    c[i] = a[i] + b[i];
  }
}

int main() {
  int size = 1 << 25;
  int block_size = 512;

  int NO_BYTES = size * sizeof(int);

  // host pointers
  int *h_a, *h_b, *gpu_results, *h_c;
  h_a = (int *)malloc(NO_BYTES);
  h_b = (int *)malloc(NO_BYTES);
  gpu_results = (int *)malloc(NO_BYTES);
  h_c = (int *)malloc(NO_BYTES);

  // initialize host pointer
  time_t t;
  srand((unsigned)time(&t));
  for (int i = 0; i < size; i++) {
    h_a[i] = (int)(rand() & 0xFF);
    h_b[i] = (int)(rand() & 0xFF);
  }

  // summation in CPU
  clock_t cpu_start, cpu_end;
  cpu_start = clock();
  sum_array_cpu(h_a, h_b, h_c, size);
  cpu_end = clock();

  memset(gpu_results, 0, NO_BYTES);

  // device pointer
  int *d_a, *d_b, *d_c;
  gpuErrchk(cudaMalloc((int **)&d_a, NO_BYTES));
  gpuErrchk(cudaMalloc((int **)&d_b, NO_BYTES));
  gpuErrchk(cudaMalloc((int **)&d_c, NO_BYTES));

  clock_t htod_start, htod_end;
  htod_start = clock();
  cudaMemcpy(d_a, h_a, NO_BYTES, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, h_b, NO_BYTES, cudaMemcpyHostToDevice);
  htod_end = clock();

  // launching the grid
  dim3 block(block_size);
  dim3 grid((size + block.x - 1) / block.x);

  // execution time mesuring in GPU
  clock_t gpu_start, gpu_end;
  gpu_start = clock();
  sum_array_gpu<<<grid, block>>>(d_a, d_b, d_c, size);
  cudaDeviceSynchronize();
  gpu_end = clock();

  // memory transfer back to host
  clock_t dtoh_start, dtoh_end;
  dtoh_start = clock();
  cudaMemcpy(gpu_results, d_c, NO_BYTES, cudaMemcpyDeviceToHost);
  dtoh_end = clock();

  // array comparison
  compare_arrays(gpu_results, h_c, size);

  printf("Sum array CPU execution time : % 4.6f \n",
         (double)((double)(cpu_end - cpu_start) / CLOCKS_PER_SEC));
  printf("Sum array GPU execution time : % 4.6f \n",
         (double)((double)(gpu_end - gpu_start) / CLOCKS_PER_SEC));
  printf("htod mem transfer time : % 4.6f \n",
         (double)((double)(htod_end - htod_start) / CLOCKS_PER_SEC));
  printf("dtoh mem transfer time : % 4.6f \n",
         (double)((double)(dtoh_end - dtoh_start) / CLOCKS_PER_SEC));
  printf("Sum array GPU total execution time : % 4.6f \n",
         (double)((double)(dtoh_end - htod_start) / CLOCKS_PER_SEC));

  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);

  free(h_a);
  free(h_b);
  free(gpu_results);

  cudaDeviceReset();

  return 0;
}