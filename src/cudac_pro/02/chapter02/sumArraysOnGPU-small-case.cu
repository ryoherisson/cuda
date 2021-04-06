#include <cuda_runtime.h>
#include <stdio.h>

#define CHECK(call)                                                         \
{                                                                           \
    const cudaError_t error = call;                                         \
    if (error != cudaSuccess)                                               \
    {                                                                       \
        printf("Error: %s:%d, ", __FILE__, __LINE__);                       \
        printf("code:%d, reason:%s\n", error, cudaGetErrorString(error));   \
        exit(1);                                                            \
    }                                                                       \
}

void checkResult(float *hostRef, float *gpuRef, const int N) {
    double epsilon = 1.0E-8;
    bool match = 1;

    for (int i = 0; i < N; i++) {
        if (abs(hostRef[i] - gpuRef[i]) > epsilon) {
            match = 0;
            printf("Arrays do not match!\n");
            printf("host %5.2f gpu %5.2f at current %d\n, ", hostRef[i], gpuRef[i], i);
            break;
        }
    }

    if (match) printf("Arrays match.\n");

    return;
}

void initialData(float *ip, int size) {
    time_t t;
    srand((unsigned) time(&t));

    for (int i = 0; i < size; i++) {
        ip[i] = (float)(rand() & 0xFF) / 10.0f;
    }
}

void sumArraysOnHost(float *A, float *B, float *C, const int N) {
    for (int idx=0; idx<N; idx++) {
        C[idx] = A[idx] + B[idx];
    }
}

__global__ void sumArraysOnGPU(float *A, float *B, float *C) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    C[i] = A[i] + B[i];
}

int main(int argc, char **argv) {
    printf("%s Starting...\n", argv[0]);

    // setup device
    int dev = 0;
    cudaSetDevice(dev);

    // set vector size
    int nElem = 32;
    printf("Vector size %d\n", nElem);

    // host memory
    size_t nBytes = nElem * sizeof(float);

    float *h_A, *h_B, *hostRef, *gpuRef;
    h_A = (float *)malloc(nBytes);
    h_B = (float *)malloc(nBytes);
    hostRef = (float *)malloc(nBytes);
    gpuRef = (float *)malloc(nBytes);

    // initialize data on host
    initialData(h_A, nElem);
    initialData(h_B, nElem);

    memset(hostRef, 0, nBytes);
    memset(gpuRef, 0, nBytes);

    // global memory on device
    float *d_A, *d_B, *d_C;
    cudaMalloc((float **)&d_A, nBytes);
    cudaMalloc((float **)&d_B, nBytes);
    cudaMalloc((float **)&d_C, nBytes);

    // host to device
    cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, nBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, gpuRef, nBytes, cudaMemcpyHostToDevice);

    // kernel on Host
    dim3 block(nElem);
    dim3 grid(1);


    sumArraysOnGPU<<< grid, block >>> (d_A, d_B, d_C);
    printf("Execution configure <<<%d, %d>>>\n", grid.x, block.x);

    // copy kernel result to host
    cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost);

    // sum arrays on Host
    sumArraysOnHost(h_A, h_B, hostRef, nElem);

    // check result
    checkResult(hostRef, gpuRef, nElem);

    // free cuda
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    
    // free host
    free(h_A);
    free(h_B);
    free(hostRef);
    free(gpuRef);

    // Reset
    cudaDeviceReset();
    return(0);
}