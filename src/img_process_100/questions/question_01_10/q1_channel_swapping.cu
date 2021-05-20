#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>

#include <iostream>

#include "../common/common.h"

__global__ void RGBToBGRDevice(uint8_t* input, const int width, const int height, const int colorWidthStep) {

    unsigned int ix = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int iy = blockIdx.y * blockDim.y + threadIdx.y;

    if ((ix < width) && (iy < height))
    {
        const int color_tid = iy * colorWidthStep + (3 * ix);
        const uint8_t t = input[color_tid + 0];

        input[color_tid + 0] = input[color_tid + 2];
        input[color_tid + 2] = t;
    }

}

void RGBToBGRHost(cv::Mat& img, cv::Mat& img_out) {

    for (int i = 0; i < img.rows; ++i) {
        // Get the first pixel pointer of the i-th line
        cv::Vec3b *p1 = img.ptr<cv::Vec3b>(i);
        cv::Vec3b *p2 = img_out.ptr<cv::Vec3b>(i);

        for (int j = 0; j < img.cols; ++j) {
            p2[j][2] = p1[j][0];
            p2[j][1] = p1[j][1];
            p2[j][0] = p1[j][2];
        }
    }
}

int main(int argc, char* const argv[]) {

    // read image
    cv::Mat img_orig;
    img_orig = cv::imread("../../data/cat.jpg", 1);

    // check data
    if (!img_orig.data)
    {
        printf("No image data \n");
        return -1;
    }

    int width = img_orig.cols;
    int height = img_orig.rows;

    cv::Mat host_img_out(height, width, CV_8UC3);
    cv::Mat gpu_img_out(height, width, CV_8UC3);

    // CPU
    // execution time mesuring in CPU
    double cpu_start, cpu_end;
    cpu_start = seconds();
    RGBToBGRHost(img_orig, host_img_out);
    cv::imwrite("./images/channel_swapping_cpu.jpg", host_img_out);
    cpu_end = seconds();
    std::cout << "CPU time: " << cpu_end - cpu_start << std::endl;

    // GPU
    // malloc device global memory
    const int n_bytes = img_orig.step * height;
    uint8_t *device_img;
    CHECK(cudaMalloc((uint8_t **)&device_img, sizeof(uint8_t) * n_bytes));

    // transfer data from host to device
    CHECK(cudaMemcpy(device_img, img_orig.data, sizeof(uint8_t) * n_bytes, cudaMemcpyHostToDevice));

    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

    // execution time mesuring in GPU
    double gpu_start, gpu_end;
    gpu_start = seconds();
    RGBToBGRDevice<<<grid, block>>>(device_img, width, height, img_orig.step);
    CHECK(cudaDeviceSynchronize());
    gpu_end = seconds();

    std::cout << "GPU time: " << gpu_end - gpu_start << std::endl;

    // copy kernel result back to host side
    CHECK(cudaMemcpy(gpu_img_out.data, device_img, sizeof(uint8_t) * n_bytes, cudaMemcpyDeviceToHost));

    // save kernel result
    cv::imwrite("./images/channel_swapping_gpu.jpg", gpu_img_out);

    // free device global memory
    CHECK(cudaFree(device_img));

    return 0;
}