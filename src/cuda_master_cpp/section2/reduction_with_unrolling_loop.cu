#include <stdio.h>
#include <stdlib.h>

#include "../common/common.h"
#include "../common/cuda_common.cuh"

#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

__global__ void reduction_unrolling_block2(int *input, int *temp, int size) {
  int tid = threadIdx.x;

  int BLOCK_OFFSET = blockIdx.x * blockDim.x * 2;

  int index = BLOCK_OFFSET + tid;

  int *i_data = input + BLOCK_OFFSET;

  if ((index + blockDim.x) < size) {
    input[index] += input[index + blockDim.x];
  }

  __syncthreads();

  for (int offset = blockDim.x / 2; offset > 0; offset = offset / 2) {
    if (tid < offset) {
      i_data[tid] += i_data[tid + offset];
    }

    __syncthreads();
  }

  if (tid == 0) {
    temp[blockIdx.x] = i_data[0];
  }
}

__global__ void reduction_unrolling_block4(int *input, int *temp, int size) {
  int tid = threadIdx.x;

  int BLOCK_OFFSET = blockIdx.x * blockDim.x * 4;

  int index = BLOCK_OFFSET + tid;

  int *i_data = input + BLOCK_OFFSET;

  if ((index + 3 * blockDim.x) < size) {
    int a1 = input[index];
    int a2 = input[index + blockDim.x];
    int a3 = input[index + 2 * blockDim.x];
    int a4 = input[index + 3 * blockDim.x];
    input[index] += a1 + a2 + a3 + a4;
  }

  __syncthreads();

  for (int offset = blockDim.x / 2; offset > 0; offset = offset / 2) {
    if (tid < offset) {
      i_data[tid] += i_data[tid + offset];
    }

    __syncthreads();
  }

  if (tid == 0) {
    temp[blockIdx.x] = i_data[0];
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
  dim3 grid((size + block.x - 1) / block.x / 4);

  printf("Kernel launch parameters | grid.x: %d, block.x: %d \n", grid.x,
         block.x);

  int temp_array_byte_size = sizeof(int) * grid.x;
  h_ref = (int *)malloc(temp_array_byte_size);

  int *d_input, *d_temp;

  gpuErrchk(cudaMalloc((void **)&d_input, byte_size));
  gpuErrchk(cudaMalloc((void **)&d_temp, byte_size));

  gpuErrchk(cudaMemset(d_temp, 0, temp_array_byte_size));
  gpuErrchk(cudaMemcpy(d_input, h_input, byte_size, cudaMemcpyHostToDevice));

  reduction_unrolling_block4<<<grid, block>>>(d_input, d_temp, size);

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