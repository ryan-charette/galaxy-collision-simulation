#include <cuda_runtime.h>

__global__ void smoke_kernel(int* out) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *out = 42;
    }
}

extern "C" int cuda_smoke_value() {
    int* device_value = nullptr;
    int host_value = 0;
    cudaMalloc(&device_value, sizeof(int));
    smoke_kernel<<<1, 32>>>(device_value);
    cudaMemcpy(&host_value, device_value, sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(device_value);
    return host_value;
}
