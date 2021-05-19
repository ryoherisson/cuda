#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <opencv2/opencv.hpp>
#include <iostream>

__global__ void convertToGray(uchar3 *color_pixel, unsigned char* gray_pixel) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    gray_pixel[idx] = (unsigned char)(0.299f*color_pixel[idx].x)
                     + 0.587f * (float)color_pixel[idx].y
                     + 0.114f * (float)color_pixel[idx].z;
}

int main(int argc, char* argv)
{
    // read image
    cv::Mat input_img = cv::imread("sample.jpg", 1);
    if (input_img.empty() == true)
    {
        std::cerr << "Error : cannnot find input image" << std::endl;
    }

    // image size
    int width = input_img.cols;
    int height = input_img.rows;
    std::cout << "Image size: " << width << "x" << height << std::endl;

    // host array
    uchar3* host_img_array_color = new uchar3[width * height];
    unsigned char* host_img_array_gray = new unsigned char [width * height];

    // to 1 array
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            host_img_array_color[x + y * width]
             = make_uchar3(input_img.at<cv::Vec3b>(y, x)[2], input_img.at<cv::Vec3b>(y, x)[1], input_img.at<cv::Vec3b>(y, x)[0]);
        }
    }

    // GPU memory
    uchar3* device_img_array_color;
    unsigned char* device_img_array_gray;
    int datasize_color = sizeof(uchar3) * width * height;
    int datasize_gray = sizeof(unsigned char) * width * height;
    cudaMalloc((void**)&device_img_array_color, datasize_color);
    cudaMalloc((void**)&device_img_array_gray, datasize_gray);

    // CPU to GPU
    cudaMemcpy(device_img_array_color, host_img_array_color, datasize_color, cudaMemcpyHostToDevice);

    // GPU
    convertToGray << <width * height, 1 >> > (device_img_array_color, device_img_array_gray);

    // GPU to CPU
    cudaMemcpy(host_img_array_gray, device_img_array_gray, datasize_gray, cudaMemcpyDeviceToHost);

    // Results
    cv::Mat1b output_img(height, width);
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            output_img.at<unsigned char>(y, x) = host_img_array_gray[x + y * width];
        }
    }
    cv::imwrite("test_gray.jpg", output_img);

    cudaFree(device_img_array_color);
    cudaFree(device_img_array_gray);
    delete host_img_array_color;
    delete host_img_array_gray;

    return 0;
}