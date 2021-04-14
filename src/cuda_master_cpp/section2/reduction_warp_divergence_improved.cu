#include <stdio.h>
#include <stdlib.h>

#include "../common/common.h"
#include "../common/cuda_common.cuh"

#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

__global__ void reduction_neighbored_pairs_improved(int *int_array,
                                                    int *temp_array, int size) {

  int tid = threadIdx.x;
  int gid = blockDim.x * blockIdx.x + threadIdx.x;

  // local data block pointer
  int *i_data = int_array + blockDim.x * blockIdx.x;

  if (gid > size) {
    return;
  }

  for (int offset = 1; offset <= blockDim.x / 2; offset *= 2) {
    int index = 2 * offset * tid;

    if (index < blockDim.x) {
      i_data[index] += i_data[index + offset];
    }

    __syncthreads();
  }

  if (tid == 0) {
    temp_array[blockIdx.x] = int_array[gid];
  }
}

int main(int argc, char **argv) {

  printf("Running neighbored pairs reduction kernel \n");

  int size = 1 << 27; // 128Mb data
  int byte_size = size * sizeof(int);
  int block_size = 128;

  int *h_input, *h_ref;
  h_input = (int *)malloc(byte_size);

  initialize(h_input, size, INIT_RANDOM);

  // get the reduction result from cpu
  int cpu_result = reduction_cpu(h_input, size);

  dim3 block(block_size);
  dim3 grid((size + block.x - 1) / block.x);

  printf("Kernel launch parameters | grid.x: %d, block.x: %d \n", grid.x,
         block.x);

  int temp_array_byte_size = sizeof(int) * grid.x;
  h_ref = (int *)malloc(temp_array_byte_size);

  int *d_input, *d_temp;

  gpuErrchk(cudaMalloc((void **)&d_input, byte_size));
  gpuErrchk(cudaMalloc((void **)&d_temp, byte_size));

  gpuErrchk(cudaMemset(d_temp, 0, temp_array_byte_size));
  gpuErrchk(cudaMemcpy(d_input, h_input, byte_size, cudaMemcpyHostToDevice));

  reduction_neighbored_pairs_improved<<<grid, block>>>(d_input, d_temp, size);

  gpuErrchk(cudaDeviceSynchronize());

  cudaMemcpy(h_ref, d_temp, temp_array_byte_size, cudaMemcpyDeviceToHost);

  int gpu_result = 0;

  for (int i = 0; i < grid.x; i++) {
    gpu_result += h_ref[i];
  }

  // validity check
  compare_results(gpu_result, cpu_result);

  gpuErrchk(cudaFree(d_temp));
  gpuErrchk(cudaFree(d_input));

  free(h_ref);
  free(h_input);

  gpuErrchk(cudaDeviceReset());
  return 0;
  return 0;
}