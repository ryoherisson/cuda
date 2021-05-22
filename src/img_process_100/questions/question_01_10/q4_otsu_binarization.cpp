#include <iostream>
#include <opencv2/opencv.hpp>

#include "../common/common.h"

// histogram statistics
__global__ void histInDevice(unsigned char* data, int* hist, int width, int height) {
    int ix = threadIdx.x + blockIdx.x * blockDim.x;
    int iy = threadIdx.y + blockIdx.y * blockDim.y;

    if (ix < width && iy < height) {
        atomicAdd(&hist[data[iy * width + ix]], 1);
    }
}

// maximum between-class variance
__global__ void OTSUthresh(const int* hist, float* sum, float*s, float* n, float* val, int width, int height, int* OtsuThresh) {
    if (blockIdx.x == 0) {
        int idx = threadIdx.x;
        atomicAdd(&sum[0], hist[idx] * idx);
    }
    else {
        int idx = threadIdx.x;
        if (idx < blockIdx.x) {
            atomicAdd(&s[blockIdx.x - 1], hist[idx] * idx);
            atomicAdd(&n[blockIdx.x - 1], hist[idx]);
        }
    }
    __syncthreads(); // All threads are synchronized
    if (blockIdx.x > 0) {
        int idx = blockIdx.x - 1;
        float u = sum[0] / (width * height);
        float w0 = n[idx] / (width * height);
        float u0 = s[idx] / n[idx];
        if (w0 == 1) {
            val[idx] = 0;
        } else {
            val[idx] = (u - u0) * (u - u0) * w0 / (1 - w0);
        }
    }

    __syncthreads(); // All threads are synchronized
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        float maxval = 0;
        for (int i = 0; i < 256; i++) {
            if (val[i] > maxval) {
                maxval = val[i];
                OtsuThresh[0] = i;
                OtsuThresh[1] = val[i];
            }
        }
    }
}

// thresholding
__global__ void OTSUInDevice(unsigned char* data_in, unsigned char* data_out, int width, int height, int* h_thresh) {
    int ix = threadIdx.x + blockIdx.x * blockDim.x;
    int iy = threadIdx.y + blockIdx.y * blockDim.y;

    if (ix < width && iy < height) {
        if (data_in[iy * width + ix] > h_thresh[0]) {
            data_out[iy * width + ix] = 255;
        }
    }
}

int main(int argc, char* const argv[]) {

    // read image with gray
    cv::Mat img_orig;
    img_orig = cv::imread("../../data/cat.jpg", 0);

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
    cv::threshold(img_orig, host_img_out, 0, 255, cv::THRESH_OTSU);
    cv::imwrite("./images/q4_otsu_binarization_cpu.jpg", host_img_out);
    cpu_end = seconds();
    std::cout << "CPU time: " << cpu_end - cpu_start << std::endl;


    // OpenCV CUDA
    // => THRESH_OTSU is not supported

    // GPU
    // malloc device global memory
    const int n_bytes = width * height;
    unsigned char* d_in;
    int* d_hist;
    CHECK(cudaMalloc((void **)&d_in, sizeof(unsigned char) * n_bytes));
    CHECK(cudaMalloc((void **)&d_hist, 256 * sizeof(int)));

    // transfer data from host to device
    CHECK(cudaMemcpy(d_in, img_orig.data, sizeof(unsigned char) * n_bytes, cudaMemcpyHostToDevice));

    // Otsu Threshold
    float* d_sum;
    float* d_s;
    float* d_n;
    float* d_val;
    int* d_thresh;
    unsigned char* d_out;

    CHECK(cudaMalloc((void **)&d_sum, sizeof(float)));
    CHECK(cudaMalloc((void **)&d_s, sizeof(float) * 256));
    CHECK(cudaMalloc((void **)&d_n, sizeof(float) * 256));
    CHECK(cudaMalloc((void **)&d_val, sizeof(float) * 256));
    CHECK(cudaMalloc((void **)&d_thresh, sizeof(int) * 2));
    CHECK(cudaMalloc((void **)&d_out, sizeof(unsigned char) * n_bytes));

    dim3 block1(32, 32);
    dim3 grid1((width + block1.x - 1) / block1.x, (height + block1.y - 1) / block1.y);

    dim3 block2(256, 1);
    dim3 grid2(257, 1);

    // execution time mesuring in GPU
    double gpu_start, gpu_end;
    gpu_start = seconds();
    histInDevice<<<grid1, block1>>>(d_in, d_hist, width, height);
    CHECK(cudaDeviceSynchronize());

    OTSUthresh<<<grid2, block2>>>(d_hist, d_sum, d_s, d_n, d_val, width, height, d_thresh);
    OTSUInDevice <<<grid1, block1>>>(d_in, d_out, width, height, d_thresh);

    gpu_end = seconds();

    std::cout << std::fixed << std::setprecision(6) << "GPU time: " << gpu_end - gpu_start << std::endl;

    // copy kernel result back to host side
    CHECK(cudaMemcpy(gpu_img_out.data, d_out, sizeof(unsigned char) * n_bytes, cudaMemcpyDeviceToHost));

    // save kernel result
    cv::imwrite("./images/q4_otsu_binarization_gpu.jpg", gpu_img_out);

    // // free device global memory
    CHECK(cudaFree(d_in));
    CHECK(cudaFree(d_out));
    CHECK(cudaFree(d_hist));
    CHECK(cudaFree(d_sum));
    CHECK(cudaFree(d_s));
    CHECK(cudaFree(d_n));
    CHECK(cudaFree(d_val));
    CHECK(cudaFree(d_thresh));

    return 0;
}