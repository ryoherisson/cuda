#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>
#include <stdio.h>
#include <time.h>

// GPU function
__global__ void gpu_function(float *d_x, float *d_y)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    d_y[i] = sin(d_x[i]) * sin(d_x[i]) + cos(d_x[i]) * cos(d_x[i]);
}

// CPU function
void cpu_function(int n, float *x, float *y)
{
    for (int i = 0; i < n; i++)
    {
        y[i] = sin(x[i]) * sin(x[i]) + cos(x[i]) * cos(x[i]);
    }
}

int main(void)
{

    int N = 10000000;
    float *host_x, *host_y, *dev_x, *dev_y, *gpu_y;

    // CPU memory
    host_x = (float*)malloc(N * sizeof(float));
    host_y = (float*)malloc(N * sizeof(float));
    gpu_y = (float*)malloc(N * sizeof(float));

    // random
    for (int i = 0; i < N; i++)
    {
        host_x[i] = rand();
    }

    // CPU
    int cpu_start = clock();

    // CPU calculation
    cpu_function(N, host_x, host_y);

    int cpu_end = clock();

    // GPU
    int gpu_start = clock();

    // Device memory
    cudaMalloc(&dev_x, N * sizeof(float));
    cudaMalloc(&dev_y, N * sizeof(float));

    // CPU to Device
    cudaMemcpy(dev_x, host_x, N * sizeof(float), cudaMemcpyHostToDevice);

    // GPU calculation
    gpu_function<<<(N + 255) / 256, 256 >>>(dev_x, dev_y);

    // GPU to CPU
    cudaMemcpy(gpu_y, dev_y, N * sizeof(float), cudaMemcpyDeviceToHost);

    int gpu_end = clock();

    // Check result
    float cpu_sum = 0.0f;
    float gpu_sum = 0.0f;

    for (int j = 0; j < N; j++)
    {
        cpu_sum += host_y[j];
        gpu_sum += gpu_y[j];
    }

    printf("CPU sum: %f, GPU sum: %f \n", cpu_sum, gpu_sum);
    printf("cpu time: %d, gpu time: %d \n", cpu_end - cpu_start, gpu_end - gpu_start);

    free(host_x);
    free(host_y);
    free(gpu_y);

    cudaFree(dev_x);
    cudaFree(dev_y);

    return 0;
}