#include "kdtree_gpu.h"
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void sided_distance_forward_cuda_kernel_2d(
    int b, int n, const scalar_t *xyz,
    int m, const scalar_t *xyz2,
    scalar_t *result, int64_t *result_i)
{
  const int batch = 512;
  constexpr int dim = 2;
  __shared__ scalar_t buf[batch * dim];

  for (int i = blockIdx.x; i < b; i += gridDim.x)
  {
    for (int k2 = 0; k2 < m; k2 += batch)
    {
    
      int end_k = min(m, k2 + batch) - k2;

      for (int j = threadIdx.x; j < end_k * dim; j += blockDim.x)
      {
        buf[j] = xyz2[(i * m + k2) * dim + j];
      }

      __syncthreads();

      for (int j = threadIdx.x + blockIdx.y * blockDim.x; j < n; j += blockDim.x * gridDim.y)
      {
        scalar_t x1 = xyz[(i * n + j) * dim + 0];
        scalar_t y1 = xyz[(i * n + j) * dim + 1];

        int64_t best_i = 0;
        scalar_t best = 0;
        int end_ka = end_k - (end_k & dim);

        best = 10000.0;
        for (int k = 0; k < end_k; k++)
        {
          scalar_t x2 = buf[k * dim + 0] - x1;
          scalar_t y2 = buf[k * dim + 1] - y1;
          scalar_t d = x2 * x2 + y2 * y2;

          if (d < best)
          {
            best = d;
            best_i = k + k2;
          }
        }

        if (k2 == 0 || result[(i * n + j)] > best)
        {
          result[(i * n + j)] = best;
          result_i[(i * n + j)] = best_i;
        }
      }
      __syncthreads();
    }
  }
}

void SearchGpu::set_param(int src_size, int tar_size)
{
  src_size_ = src_size;
  tar_size_ = tar_size;
  cudaMallocManaged(&dist_, src_size_ * sizeof(float));
  cudaMallocManaged(&idx_, src_size_ * sizeof(int64_t));
  cudaMallocManaged(&src_, src_size_ * 2 * sizeof(float));
  cudaMallocManaged(&tar_, tar_size_ * 2 * sizeof(float));
}

std::vector<int> SearchGpu::kdsearch_gpu(const std::vector<PointT> &src, const std::vector<PointT> &tar)
{
  for (int i = 0; i < src.size(); i++)
  {
    src_[i * 2] = src[i][0];
    src_[i * 2 + 1] = src[i][1];
  }

  for (int i = 0; i < tar.size(); i++)
  {
    tar_[i * 2] = tar[i][0];
    tar_[i * 2 + 1] = tar[i][1];
  }

  int b = 1;
  sided_distance_forward_cuda_kernel_2d<float><<<dim3(32, 16, 1), 512, 0>>>(b, src_size_, src_, tar_size_, tar_, dist_, idx_);
  cudaDeviceSynchronize();

  std::vector<int> result(src.size());
  for (int i = 0; i < src_size_; i++)
  {
    result[i] = idx_[i];
  }
  return result;
}

void SearchGpu::release()
{
  cudaFree(src_);
  cudaFree(tar_);
  cudaFree(idx_);
  cudaFree(dist_);
}