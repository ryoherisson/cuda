#include <iostream>
#include <opencv2/opencv.hpp>

#include "../common/common.h"

__global__ void binarize(uchar3* color, unsigned char* gray, int threashold) {

    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    gray[idx] = (unsigned char)(0.299f*(float)color[idx].x
                + 0.587f * (float)color[idx].y
                + 0.114f * (float)color[idx].z) / threashold;

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
    cv::Mat cvcuda_img_out(height, width, CV_8UC1);
    cv::Mat gpu_img_out(height, width, CV_8UC1);

    // threshold
    int threashold = 127;

    // CPU
    // execution time mesuring in CPU
    cv::Mat host_gray(height, width, CV_8UC1);

    double cpu_start, cpu_end;
    cpu_start = seconds();
    cv::cvtColor(img_orig, host_gray, cv::COLOR_BGR2GRAY);
    cv::threshold(host_gray, host_img_out, threashold, 255, cv::THRESH_BINARY);
    cv::imwrite("./images/q3_binarization_cpu.jpg", host_img_out);
    cpu_end = seconds();
    std::cout << "CPU time: " << cpu_end - cpu_start << std::endl;


    // OpenCV CUDA
    cv::cuda::GpuMat dst, mid, src;
    src.upload(img_orig);

    double cvcuda_start, cvcuda_end;
    cvcuda_start = seconds();
    cv::cuda::cvtColor(src, mid, cv::COLOR_BGR2GRAY);
    cv::cuda::threshold(mid, dst, threashold, 255, cv::THRESH_BINARY);
    cvcuda_end = seconds();

    std::cout << "OpenCV CUDA time: " << cvcuda_end - cvcuda_start << std::endl;

    dst.download(cvcuda_img_out);

    // save kernel result
    cv::imwrite("./images/q3_binarization_cvcuda.jpg", cvcuda_img_out);


    // GPU
    // host array
    uchar3* host_img_color = new uchar3 [width * height];
    unsigned char* host_img_binary = new unsigned char [width * height];

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
    unsigned char* device_img_binary;
    CHECK(cudaMalloc((void **)&device_img_color, sizeof(uchar3) * n_bytes));
    CHECK(cudaMalloc((void **)&device_img_binary, sizeof(unsigned char) * n_bytes));

    // transfer data from host to device
    CHECK(cudaMemcpy(device_img_color, host_img_color, sizeof(uchar3) * n_bytes, cudaMemcpyHostToDevice));

    dim3 block(32);
    dim3 grid((width * height + block.x - 1) / block.x);

    // execution time mesuring in GPU
    double gpu_start, gpu_end;
    gpu_start = seconds();
    binarize<<<grid, block>>>(device_img_color, device_img_binary, threashold);
    CHECK(cudaDeviceSynchronize());
    gpu_end = seconds();

    std::cout << std::fixed << std::setprecision(6) << "GPU time: " << gpu_end - gpu_start << std::endl;

    // copy kernel result back to host side
    CHECK(cudaMemcpy(host_img_binary, device_img_binary, sizeof(unsigned char) * n_bytes, cudaMemcpyDeviceToHost));

    // Results
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            gpu_img_out.at<unsigned char>(y, x) = host_img_binary[x + y * width];
        }
    }

    // save kernel result
    cv::imwrite("./images/q3_binarization_gpu.jpg", gpu_img_out);

    // free device global memory
    CHECK(cudaFree(device_img_color));
    CHECK(cudaFree(device_img_binary));

    return 0;
}