#include <stdio.h>

__global__ void helloFromGPU()
{
    if (threadIdx.x == 5)
    {
        printf("Hello World from GPU! %d\n", threadIdx.x);
    }
}

int main(int argc, char **argv)
{
    printf("Hello world from CPU!\n");

    helloFromGPU<<<1, 10>>>();
    cudaDeviceReset();
    return 0;
}