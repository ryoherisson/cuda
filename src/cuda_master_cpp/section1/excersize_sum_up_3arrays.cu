/* execution time summary
[GeForce RTX 2070 SUPER]

Sum array CPU vs GPU execution time
(Block size : CPU vs GPU)
64:  0.015141 vs 0.000244
128: 0.015224 vs 0.000245
256: 0.015389 vs 0.000248
512: 0.015349 vs 0.000246

To run the code, add block size as argument
$ ./a.out 128
*/

// #include "cuda_common.cuh"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <time.h>
// for memset
#include <cstring>

#include "../common/common.h"
#include "../common/cuda_common.cuh"

__global__ void sum_array_gpu(int *a, int *b, int *c, int *d, int size) {
  int gid = blockIdx.x * blockDim.x + threadIdx.x;

  if (gid < size) {
    d[gid] = a[gid] + b[gid] + c[gid];
  }
}

void sum_array_cpu(int *a, int *b, int *c, int *d, int size) {
  for (int i = 0; i < size; i++) {
    d[i] = a[i] + b[i] + c[i];
  }
}

int main(int argc, char *argv[]) {
  int size = 1 << 22;
  printf("size: %d \n", size);
  int block_size = atoi(argv[1]);

  int NO_BYTES = size * sizeof(int);

  // host pointers
  int *h_a, *h_b, *h_c, *h_d, *gpu_results;
  h_a = (int *)malloc(NO_BYTES);
  h_b = (int *)malloc(NO_BYTES);
  h_c = (int *)malloc(NO_BYTES);
  gpu_results = (int *)malloc(NO_BYTES);
  h_d = (int *)malloc(NO_BYTES);

  // initialize host pointer
  time_t t;
  srand((unsigned)time(&t));
  for (int i = 0; i < size; i++) {
    h_a[i] = (int)(rand() & 0xFF);
    h_b[i] = (int)(rand() & 0xFF);
    h_c[i] = (int)(rand() & 0xFF);
  }

  // summation in CPU
  clock_t cpu_start, cpu_end;
  cpu_start = clock();
  sum_array_cpu(h_a, h_b, h_c, h_d, size);
  cpu_end = clock();

  memset(gpu_results, 0, NO_BYTES);

  // device pointer
  int *d_a, *d_b, *d_c, *d_d;
  gpuErrchk(cudaMalloc((int **)&d_a, NO_BYTES));
  gpuErrchk(cudaMalloc((int **)&d_b, NO_BYTES));
  gpuErrchk(cudaMalloc((int **)&d_c, NO_BYTES));
  gpuErrchk(cudaMalloc((int **)&d_d, NO_BYTES));

  clock_t htod_start, htod_end;
  htod_start = clock();
  gpuErrchk(cudaMemcpy(d_a, h_a, NO_BYTES, cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(d_b, h_b, NO_BYTES, cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(d_c, h_c, NO_BYTES, cudaMemcpyHostToDevice));
  htod_end = clock();

  // launching the grid
  dim3 block(block_size);
  dim3 grid((size + block.x - 1) / block.x);

  // execution time mesuring in GPU
  clock_t gpu_start, gpu_end;
  gpu_start = clock();
  sum_array_gpu<<<grid, block>>>(d_a, d_b, d_c, d_d, size);
  gpuErrchk(cudaDeviceSynchronize());
  gpu_end = clock();

  // memory transfer back to host
  clock_t dtoh_start, dtoh_end;
  dtoh_start = clock();
  gpuErrchk(cudaMemcpy(gpu_results, d_d, NO_BYTES, cudaMemcpyDeviceToHost));
  dtoh_end = clock();

  // array comparison
  compare_arrays(gpu_results, h_d, size);

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

  gpuErrchk(cudaFree(d_a));
  gpuErrchk(cudaFree(d_b));
  gpuErrchk(cudaFree(d_c));
  gpuErrchk(cudaFree(d_d));

  free(h_a);
  free(h_b);
  free(h_c);
  free(h_d);
  free(gpu_results);

  cudaDeviceReset();

  return 0;
}