#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <cstring>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

__global__ void mem_trs_3d(int *input) {
  int tid = threadIdx.z * blockDim.x * blockDim.y + blockDim.x * threadIdx.y +
            threadIdx.x;
  int num_threads_in_a_block = blockDim.x * blockDim.y * blockDim.z;
  int blockOffset = blockIdx.x * num_threads_in_a_block;

  int num_threads_in_a_row = num_threads_in_a_block * gridDim.x;
  int row_offset = blockIdx.y * num_threads_in_a_row;

  int num_threads_in_a_depth = num_threads_in_a_row * gridDim.y;
  int depth_offset = blockIdx.z * num_threads_in_a_depth;

  int gid = tid + blockOffset + row_offset + depth_offset;

  //   printf("value: %d \n", input[gid]);
  printf("blockIdx.x: %d, blockIdx.y: %d, blockIdx.z: %d, threadIdx.x: %d, "
         "gid: %d, - data: %d \n",
         blockIdx.x, blockIdx.y, blockIdx.z, tid, gid, input[gid]);
}

int main() {
  int size = 64;
  int byte_size = size * sizeof(int);

  int *h_input;
  h_input = (int *)malloc(byte_size);

  time_t t;
  srand((unsigned)time(&t));
  for (int i = 0; i < size; i++) {
    h_input[i] = (int)(rand() & 0xff);
  }

  int *d_input;
  cudaMalloc((void **)&d_input, byte_size);

  cudaMemcpy(d_input, h_input, byte_size, cudaMemcpyHostToDevice);

  dim3 block(2, 2, 2);
  dim3 grid(2, 2, 2);

  mem_trs_3d<<<grid, block>>>(d_input);
  cudaDeviceSynchronize();

  cudaFree(d_input);
  free(h_input);

  cudaDeviceReset();

  return 0;
}