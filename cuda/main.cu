#include <iostream>
#include <vector>
#include <cuda_runtime.h>

template <typename T>
struct CudaAllocator{
    using value_type = T;

    T *allocate(size_t size)
    {
        T *ptr = nullptr;
        cudaMallocManaged(&ptr, size * sizeof(T));
        return ptr;
    }

    void deallocate(T *ptr, size_t size = 0)
    {
        cudaFree(ptr);
    }
};


__device__ __host__ __inline__ void say_hello()
{
#ifdef __CUDA_ARCH__
    printf("device hello cuda %d %d %d\n", __CUDA_ARCH__, threadIdx.x, blockDim.x);
#else
    printf("device hello cpu\n");
#endif
}

__global__ void kernel()
{
    say_hello();
}

template <typename Func>
__global__ void parallel_for(int n, Func func)
{
    for (int i = 0; i < n; i++)
        func(i);
}

int main(int, char**) {
    constexpr int n = 100;
    std::vector<float, CudaAllocator<float>> arr(n);
    parallel_for<<<1, 1>>>(n, [arr = arr.data()] __device__ (int i)
    {
        arr[i] = sinf(i);
    });
    cudaDeviceSynchronize();
    for (int i = 0; i < n; i++)
    {
        printf("%f\n", arr[i]);
    }
}
