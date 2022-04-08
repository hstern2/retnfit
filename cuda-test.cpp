#include <chrono>
#include <iostream>

__global__ void _cuda_parallel_multiplication(int count, int* test_data, int magnitude);

int main() {
    int count = 60000000;    // 60 million elements
    int* test_data = new int[count];

    for(int i = 0; i < count; i++)
        test_data[i] = i;

    // Perform calculation on host CPU
    auto t1 = std::chrono::high_resolution_clock::now();
    for(int i = 0; i < count; i++)
        test_data[i] = test_data[i] * 5;
    auto t2 = std::chrono::high_resolution_clock::now();
    
    // Copy data to device
    int* d_test_data;
    cudaMalloc(&d_test_data, count * sizeof(int));
    cudaMemcpy(d_test_data, test_data, count * sizeof(int), cudaMemcpyHostToDevice);
    
    // Launch kernel
    _cuda_parallel_multiplication<<<10, 1024>>>(count, d_test_data, 5);
    
    // Copy results back to device
    cudaDeviceSynchronize();
    cudaMemcpy(test_data, d_test_data, count * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_test_data);

    for(int i = 0; i < 10; i++)
        std::cout << i << ": " << test_data[i] << std::endl;

    std::cout	<< "CPU time: "
                << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count()
                << "ms" << std::endl;
}

__global__ void _cuda_parallel_multiplication(int count, int* test_data, int magnitude) {

     int globalIdx = blockIdx.x * blockDim.x + threadIdx.x;

     while (globalIdx < count) {
        test_data[globalIdx] = test_data[globalIdx] * magnitude;

        globalIdx +=  blockDim.x * gridDim.x;
        __syncthreads();
     }
}