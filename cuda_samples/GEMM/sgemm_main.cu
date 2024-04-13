#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "src/runner.cuh"
#include "src/utils.cuh"

const int ALPHA = 1;
const int BETA = 0;
const int REPEAT_TIMES = 50;
const std::vector<int> SIZE = {128, 256, 512, 1024, 2048, 4096};

int main(int argc, char **argv) {
    if (argc != 2) {
        printf("Please select a kernel version (range 0 - 3, 0 for NVIDIA cuBLAS)\n");
        exit(EXIT_FAILURE);
    }

    // get kernel_version
    int kernel_version = std::stoi(argv[1]);
    if (kernel_version < 0 || kernel_version > 3) {
        printf(
            "Please enter a valid kernel version (range 0 - 3, 0 for NVIDIA cuBLAS)\n");
        exit(EXIT_FAILURE);
    }

    // get devide_idx
    int devide_idx = 0;
    if (argc == 3) {
        devide_idx = std::stoi(argv[2]);
    }

    int M, N, K;
    float *C = nullptr, *C_cublas = nullptr;  // host matrices
    float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr,
          *d_C_cublas = nullptr;  // device matrices

    

    float elapsed_time;
    

    // run the kernel
    printf("Running kernel version %d on device %d\n", kernel_version, devide_idx);

    for (auto size : SIZE) {
        cudaErrCheck(cudaSetDevice(devide_idx));

        cublasHandle_t blas_handle;
        cublasErrCheck(cublasCreate(&blas_handle));

        cudaEvent_t start, end;
        cudaErrCheck(cudaEventCreate(&start));
        cudaErrCheck(cudaEventCreate(&end));

        M = N = K = size;

        // allocate host memory
        C = (float *)malloc(M * N * sizeof(float));
        C_cublas = (float *)malloc(M * N * sizeof(float));

        // allocate device memory
        cudaErrCheck(cudaMalloc((void **)&d_A, M * K * sizeof(float)));
        cudaErrCheck(cudaMalloc((void **)&d_B, K * N * sizeof(float)));
        cudaErrCheck(cudaMalloc((void **)&d_C, M * N * sizeof(float)));
        cudaErrCheck(cudaMalloc((void **)&d_C_cublas, M * N * sizeof(float)));

        // generate data
        curandGenerator_t generator;
        curandErrCheck(curandCreateGenerator(&generator, CURAND_RNG_PSEUDO_DEFAULT));
        curandErrCheck(curandSetPseudoRandomGeneratorSeed(generator, 1337ULL));

        curandErrCheck(curandGenerateUniform(generator, d_A, M * K));
        curandErrCheck(curandGenerateUniform(generator, d_B, K * N));

        curandErrCheck(curandDestroyGenerator(generator));

        // ----------------------------------------------------------------------------
        // verify the correctness of the kernel && warm up so as to avoid the first-time
        // overhead overhead
        // 1) self-implemented kernel
        RunSgemmKernel(kernel_version, M, N, K, ALPHA, d_A, d_B, BETA, d_C, blas_handle);
        cudaErrCheck(cudaDeviceSynchronize());
        // 2) cuBLAS
        RunSgemmKernel(0, M, N, K, ALPHA, d_A, d_B, BETA, d_C_cublas, blas_handle);
        cudaErrCheck(cudaDeviceSynchronize());
        cudaMemcpy(C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(C_cublas, d_C_cublas, M * N * sizeof(float), cudaMemcpyDeviceToHost);
        if (!IsMatrixEqual(C, C_cublas, M, N)) {
            printf("Matrix mismatch\n");

            if (size <= 128) {
                std::string err_file = "./debug/err_" + std::to_string(size) + ".txt";
                printf("Writing matrix to %s\n", err_file.c_str());
                std::ofstream out(err_file);
                out << "Matrix C: \n";
                PrintMatrix(C, M, N, out);
                out << "Matrix C_cublas: \n";
                PrintMatrix(C_cublas, M, N, out);
            }
            exit(EXIT_FAILURE);
        }

        // ----------------------------------------------------------------------------
        // measure the performance
        cudaErrCheck(cudaEventRecord(start));
        for (int i = 0; i < REPEAT_TIMES; i++) {
            RunSgemmKernel(kernel_version, M, N, K, ALPHA, d_A, d_B, BETA, d_C, blas_handle);
        }
        cudaErrCheck(cudaEventRecord(end));
        cudaErrCheck(cudaEventSynchronize(end));
        cudaErrCheck(cudaEventElapsedTime(&elapsed_time, start, end));
        elapsed_time /= 1000.0f;  // convert ms to s

        long long total_flop = static_cast<long long>(M) * N * K * REPEAT_TIMES * 2;
        double avg_elapsed_time = double(elapsed_time) / REPEAT_TIMES;
        double gflops = (double)total_flop / 1e9 / double(elapsed_time);
        printf("Size: %d, Average elapsed time: %7.6f s, Performance: %7.2f GFLOPS. \n",
               size, avg_elapsed_time, gflops);
        fflush(stdout);

        // free memory
        free(C);
        free(C_cublas);

        cudaErrCheck(cudaFree(d_A));
        cudaErrCheck(cudaFree(d_B));
        cudaErrCheck(cudaFree(d_C));
        cudaErrCheck(cudaFree(d_C_cublas));

        cublasErrCheck(cublasDestroy(blas_handle));
        cudaErrCheck(cudaEventDestroy(start));
        cudaErrCheck(cudaEventDestroy(end));

        cudaErrCheck(cudaDeviceReset());
    }

    return 0;
}