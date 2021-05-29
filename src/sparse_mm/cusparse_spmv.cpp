#include "./common/common.h"
#include <stdio.h>
#include <stdlib.h>
#include <cusparse_v2.h>
#include <cublas_v2.h>
#include <cuda.h>
#include <iostream>

//TODO: CPUとCUDAの値が一致しない


/*
 * This is an example demonstrating usage of the cuSPARSE library to perform a
 * sparse matrix-vector multiplication on randomly generated data.
 */

/*
 * M = # of rows
 * N = # of columns
 */
int M = 1024;
int N = 1024;

void checkResult(float *hostRef, float *gpuRef, const int N)
{
    double epsilon = 1.0E-8;
    bool match = 1;

    for (int i = 0; i < N; i++)
    {
        if (abs(hostRef[i] - gpuRef[i]) > epsilon)
        {
            match = 0;
            printf("index %d host %f gpu %f\n", i, hostRef[i], gpuRef[i]);
            break;
        }
    }

    if (match)
        printf("Arrays match.\n\n");
    else
        printf("Arrays do not match.\n\n");
}

/*
 * Generate a vector of length N with random single-precision floating-point
 * values between 0 and 100.
 */
void generate_random_vector(int N, float **outX)
{
    int i;
    double rMax = (double)RAND_MAX;
    float *X = (float *)malloc(sizeof(float) * N);

    for (i = 0; i < N; i++)
    {
        int r = rand();
        double dr = (double)r;
        X[i] = (dr / rMax) * 100.0;
    }

    *outX = X;
}

/*
 * Generate random dense matrix A in column-major order, while rounding some
 * elements down to zero to ensure it is sparse.
 */
int generate_random_dense_matrix(int M, int N, float **outA)
{
    int i, j;
    double rMax = (double)RAND_MAX;
    float *A = (float *)malloc(sizeof(float) * M * N);
    int totalNnz = 0;

    for (j = 0; j < N; j++)
    {
        for (i = 0; i < M; i++)
        {
            int r = rand();
            float *curr = A + (j * M + i);

            if (r % 3 > 0)
            {
                *curr = 0.0f;
            }
            else
            {
                double dr = (double)r;
                *curr = (dr / rMax) * 100.0;
            }

            if (*curr != 0.0f)
            {
                totalNnz++;
            }
        }
    }

    *outA = A;
    return totalNnz;
}

void csr_format_host(float *A, float **val, int **row_ptr, int **col_ind, int nnz, int M, int N)
{

    float *Val = (float *)malloc(sizeof(float) * nnz);
    int *ColInd = (int *)malloc(sizeof(int) * nnz);
    int *RowPtr = (int *)malloc(sizeof(int) * (N + 1));

    int nnz_ind = 0;
    RowPtr[0] = 0;

    for (int j = 0; j < N; j++)
    {
        for (int i = 0; i < M; i++)
        {
            float curr = A[j * M + i];

            if (curr != 0.0f)
            {
                Val[nnz_ind] = curr;
                ColInd[nnz_ind] = i;
                nnz_ind++;
            }
        }
        RowPtr[j+1] = nnz_ind;
    }

    *val = Val;
    *col_ind = ColInd;
    *row_ptr = RowPtr;
}


void csr_spmv_host(float *val, int *row_ptr, int *col_ind, int N, float* x, float **outY)
{
    float *Y = (float *)malloc(sizeof(float) * N);

    for (int j = 0; j < N; j++)
    {
        Y[j] = 0;
        for (int k = row_ptr[j]; k < row_ptr[j+1]; k++)
        {
            Y[j] += val[k] * x[col_ind[k]];
        }
    }
    *outY = Y;
}

int main(int argc, char **argv)
{
    int row;
    float *A, *dA;
    int *dNnzPerRow;
    float *dCsrValA;
    int *dCsrRowPtrA;
    int *dCsrColIndA;
    int totalNnz;
    float alpha = 1.0f;
    float beta = 0.0f;
    float *dX, *X;
    float *dY, *Y;
    cusparseHandle_t handle = 0;
    cusparseMatDescr_t descr = 0;

    // Generate input
    srand(9384);
    int trueNnz = generate_random_dense_matrix(M, N, &A);
    generate_random_vector(N, &X);
    generate_random_vector(M, &Y);

    // ---------------------------------
    // SpMV on Host
    // ---------------------------------
    float *hCsrVal;
    int *hCsrRowPtr;
    int *hCsrColInd;
    float *hY;

    // Calculate execution time
    double host_start, host_end;
    host_start = seconds();

    csr_format_host(A, &hCsrVal, &hCsrRowPtr, &hCsrColInd, trueNnz, M, N);
    csr_spmv_host(hCsrVal, hCsrRowPtr, hCsrColInd, N, X, &hY);

    host_end = seconds();

    // Check Result on Host
    printf("Result on Host\n");
    for (row = 0; row < 5; row++)
    {
        printf("%2.2f\n", hY[row]);
    }
    printf("\n");

    // ---------------------------------
    // SpMV on Device
    // ---------------------------------

    /*
    cuBLAS
    */
    float *dbA;
    float *dbX;
    float *bY, *dbY;
    cublasHandle_t handleb = 0;

    // generate inputs
    generate_random_vector(M, &bY);

    // Create the cuBLAS handle
    CHECK_CUBLAS(cublasCreate(&handleb));

    // Allocate device memory
    CHECK(cudaMalloc((void **)&dbA, sizeof(float) * M * N));
    CHECK(cudaMalloc((void **)&dbX, sizeof(float) * N));
    CHECK(cudaMalloc((void **)&dbY, sizeof(float) * M));

    // Transfer inputs to the device
    CHECK_CUBLAS(cublasSetVector(N, sizeof(float), X, 1, dbX, 1));
    CHECK_CUBLAS(cublasSetVector(M, sizeof(float), Y, 1, dbY, 1));
    CHECK_CUBLAS(cublasSetMatrix(M, N, sizeof(float), A, M, dbA, M));

    // Calculate execution time
    double device_b_start, device_b_end;
    device_b_start = seconds();
    // Execute the matrix-vector multiplication
    CHECK_CUBLAS(cublasSgemv(handleb, CUBLAS_OP_N, M, N, &alpha, dbA, M, dbX, 1,
                             &beta, dbY, 1));
    device_b_end = seconds();

    // Retrieve the output vector from the device
    CHECK_CUBLAS(cublasGetVector(M, sizeof(float), dbY, 1, bY, 1));
    // Check Result on Device
    printf("Result on cuBlas\n");
    for (row = 0; row < 5; row++)
    {
        printf("%2.2f\n", bY[row]);
    }

    printf("...\n");

    /*
    cuSparse
    */

    // Create the cuSPARSE handle
    CHECK_CUSPARSE(cusparseCreate(&handle));

    // Allocate device memory for vectors and the dense form of the matrix A
    CHECK(cudaMalloc((void **)&dX, sizeof(float) * N));
    CHECK(cudaMalloc((void **)&dY, sizeof(float) * M));
    CHECK(cudaMalloc((void **)&dA, sizeof(float) * M * N));
    CHECK(cudaMalloc((void **)&dNnzPerRow, sizeof(int) * M));

    // Construct a descriptor of the matrix A
    CHECK_CUSPARSE(cusparseCreateMatDescr(&descr));
    CHECK_CUSPARSE(cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL));
    CHECK_CUSPARSE(cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO));

    // Transfer the input vectors and dense matrix A to the device
    CHECK(cudaMemcpy(dX, X, sizeof(float) * N, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dY, Y, sizeof(float) * M, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dA, A, sizeof(float) * M * N, cudaMemcpyHostToDevice));

    // Compute the number of non-zero elements in A
    CHECK_CUSPARSE(cusparseSnnz(handle, CUSPARSE_DIRECTION_ROW, M, N, descr, dA,
                                M, dNnzPerRow, &totalNnz));

    if (totalNnz != trueNnz)
    {
        fprintf(stderr, "Difference detected between cuSPARSE NNZ and true "
                "value: expected %d but got %d\n", trueNnz, totalNnz);
        return 1;
    }

    // Allocate device memory to store the sparse CSR representation of A
    CHECK(cudaMalloc((void **)&dCsrValA, sizeof(float) * totalNnz));
    CHECK(cudaMalloc((void **)&dCsrRowPtrA, sizeof(int) * (M + 1)));
    CHECK(cudaMalloc((void **)&dCsrColIndA, sizeof(int) * totalNnz));

    // Calculate execution time
    double device_start, device_end;
    device_start = seconds();

    CHECK(cudaDeviceSynchronize());
    // Convert A from a dense formatting to a CSR formatting, using the GPU
    CHECK_CUSPARSE(cusparseSdense2csr(handle, M, N, descr, dA, M, dNnzPerRow,
                                      dCsrValA, dCsrRowPtrA, dCsrColIndA));

    // Perform matrix-vector multiplication with the CSR-formatted matrix A
    CHECK_CUSPARSE(cusparseScsrmv(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                  M, N, totalNnz, &alpha, descr, dCsrValA,
                                  dCsrRowPtrA, dCsrColIndA, dX, &beta, dY));

    device_end = seconds();

    // Copy the result vector back to the host
    CHECK(cudaMemcpy(Y, dY, sizeof(float) * M, cudaMemcpyDeviceToHost));

    // Check Result on Device
    printf("Result on Device\n");
    for (row = 0; row < 5; row++)
    {
        printf("%2.2f\n", Y[row]);
    }

    printf("...\n");

    printf("Exectution time Host: %.6f, cuBlas: %.6f, cuSparse: %.6f\n", host_end-host_start, device_b_end-device_b_start, device_end-device_start);

    // Check if results of host and device are same
    checkResult(hY, Y, N);

    free(A);
    free(X);
    free(Y);

    CHECK(cudaFree(dX));
    CHECK(cudaFree(dY));
    CHECK(cudaFree(dA));
    CHECK(cudaFree(dNnzPerRow));
    CHECK(cudaFree(dCsrValA));
    CHECK(cudaFree(dCsrRowPtrA));
    CHECK(cudaFree(dCsrColIndA));

    CHECK_CUSPARSE(cusparseDestroyMatDescr(descr));
    CHECK_CUSPARSE(cusparseDestroy(handle));


    return 0;
}
