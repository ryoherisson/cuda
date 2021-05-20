#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>

#include <iostream>

#include "../common/common.h"

__global__ void convertToGray(uchar3* color, unsigned char* gray) {

    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    gray[idx] = (unsigned char)(0.299f*(float)color[idx].x
                + 0.587f * (float)color[idx].y
                + 0.114f * (float)color[idx].z);

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

    cv::Mat host_img_out(height, width, CV_8UC1);
    cv::Mat gpu_img_out(height, width, CV_8UC1);

    // CPU
    // execution time mesuring in CPU
    double cpu_start, cpu_end;
    cpu_start = seconds();
    cv::cvtColor(img_orig, host_img_out, cv::COLOR_BGR2GRAY);
    cv::imwrite("./images/q2_grayscale_cpu.jpg", host_img_out);
    cpu_end = seconds();
    std::cout << "CPU time: " << cpu_end - cpu_start << std::endl;

    // GPU
    // host array
    uchar3* host_img_color = new uchar3 [width * height];
    unsigned char* host_img_gray = new unsigned char [width * height];

    // host image to 1 array
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            host_img_color[x + y * width]
             = make_uchar3(img_orig.at<cv::Vec3b>(y, x)[2], img_orig.at<cv::Vec3b>(y, x)[1], img_orig.at<cv::Vec3b>(y, x)[0]);
        }
    }

    // malloc device global memory
    const int n_bytes = width * height;
    uchar3 *device_img_color;
    unsigned char* device_img_gray;
    CHECK(cudaMalloc((void **)&device_img_color, sizeof(uchar3) * n_bytes));
    CHECK(cudaMalloc((void **)&device_img_gray, sizeof(unsigned char) * n_bytes));

    // transfer data from host to device
    CHECK(cudaMemcpy(device_img_color, host_img_color, sizeof(uchar3) * n_bytes, cudaMemcpyHostToDevice));

    dim3 block(32);
    dim3 grid((width * height + block.x - 1) / block.x);

    // execution time mesuring in GPU
    double gpu_start, gpu_end;
    gpu_start = seconds();
    convertToGray<<<grid, block>>>(device_img_color, device_img_gray);
    CHECK(cudaDeviceSynchronize());
    gpu_end = seconds();

    std::cout << "GPU time: " << gpu_end - gpu_start << std::endl;

    // copy kernel result back to host side
    CHECK(cudaMemcpy(host_img_gray, device_img_gray, sizeof(unsigned char) * n_bytes, cudaMemcpyDeviceToHost));

    // Results
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            gpu_img_out.at<unsigned char>(y, x) = host_img_gray[x + y * width];
        }
    }

    // save kernel result
    cv::imwrite("./images/q2_grayscale_gpu.jpg", gpu_img_out);

    // free device global memory
    CHECK(cudaFree(device_img_color));
    CHECK(cudaFree(device_img_gray));

    return 0;
}