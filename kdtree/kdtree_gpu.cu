#include "kdtree_gpu.h"
#include <cuda_runtime.h>

__global__ void kernel(float *data_src, int src_size, 
    float *data_tar, int tar_size, float *res)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // res += i * tar_size;
    auto x = data_src[i * 2];
    auto y = data_src[i * 2 + 1];
    for (int j = 0; j < tar_size; j++)
    {
        auto x1 = data_tar[j * 2];
        auto y1 = data_tar[j * 2 + 1];
        res[i * tar_size + j] = (x - x1) * (x - x1) + (y - y1) * (y - y1);
    }
}


std::vector<PointT> kdsearch_gpu(const std::vector<PointT>& src, const std::vector<PointT>& tar)
{
    float *res;
    float *src_v;
    float *tar_v;

    cudaMallocManaged(&res, src.size() * tar.size() * sizeof(float));
    cudaMallocManaged(&src_v, src.size() * 2 * sizeof(float));
    cudaMallocManaged(&tar_v, tar.size() * 2 * sizeof(float));

    for (int i = 0; i < src.size(); i++)
    {
        src_v[i * 2] = src[i][0];
        src_v[i * 2 + 1] = src[i][1];
    }

    for (int i = 0; i < tar.size(); i++)
    {
        tar_v[i * 2] = tar[i][0];
        tar_v[i * 2 + 1] = tar[i][1];
    }


    dim3 threads = dim3(256);
    dim3 blocks = dim3(src.size() / threads.x);
    kernel<<<blocks, threads>>>(src_v, src.size(), tar_v, tar.size(), res);
    cudaDeviceSynchronize();

    std::vector<PointT> result(src.size(), {0.0, 0.0});
    for (int i = 0; i < src.size(); i++)
    {
        int best_idx = 0;
        float min_dist = res[0];
        for (int j = 0; j < tar.size(); j++)
        {
            auto dist = res[i * tar.size() + j];
            if (dist < min_dist)
            {
                min_dist = dist;
                best_idx = j;
            }
        }
        result[i] = tar[best_idx];
    }

    cudaFree(src_v);
    cudaFree(tar_v);
    cudaFree(res);
    return result;
}

template<typename scalar_t>
__global__ void sided_distance_forward_cuda_kernel(
    int b, int n, const scalar_t * xyz,
    int m, const scalar_t * xyz2,
    scalar_t * result, int64_t * result_i) {
  const int batch=512;
  __shared__ scalar_t buf[batch*3];

  for (int i = blockIdx.x; i<b; i += gridDim.x){
    for (int k2 = 0; k2 < m; k2 += batch) {

      int end_k =  min(m, k2 + batch) - k2;

      for (int j = threadIdx.x; j < end_k * 3; j += blockDim.x) {
        buf[j]=xyz2[(i*m+k2)*3+j];
      }

      __syncthreads();

      for (int j = threadIdx.x + blockIdx.y * blockDim.x; j < n; j += blockDim.x * gridDim.y) {
        scalar_t x1 = xyz[(i * n + j) * 3 + 0];
        scalar_t y1 = xyz[(i * n + j) * 3 + 1];
        scalar_t z1 = xyz[(i * n + j) * 3 + 2];

        int64_t best_i = 0;
        scalar_t best = 0;
        int end_ka = end_k - (end_k & 3);

        if (end_ka == batch){
          for (int k = 0; k < batch; k += 4) {
            {
            scalar_t x2 = buf[k * 3 + 0] - x1;
            scalar_t y2 = buf[k * 3 + 1] - y1;
            scalar_t z2 = buf[k * 3 + 2]- z1;
            scalar_t d = x2 * x2 + y2 * y2 + z2 * z2;

            if (k == 0 || d < best) {
              best = d;
              best_i = k + k2;
            }
            }

            {
            scalar_t x2 = buf[k * 3 + 3] - x1;
            scalar_t y2 = buf[k * 3 + 4] - y1;
            scalar_t z2 = buf[k * 3 + 5] - z1;
            scalar_t d = x2 * x2 + y2 * y2 + z2 * z2;

            if (d < best){
              best = d;
              best_i = k + k2 + 1;
            }
            }

            {
            scalar_t x2 = buf[k * 3 + 6]- x1;
            scalar_t y2 = buf[k * 3 + 7] - y1;
            scalar_t z2 = buf[k * 3 + 8] - z1;
            scalar_t d = x2 * x2 + y2 * y2 + z2 * z2;

            if (d < best) {
              best = d;
              best_i = k + k2 + 2;
            }
            }

            {
            scalar_t x2 = buf[k * 3 + 9] - x1;
            scalar_t y2 = buf[k * 3 + 10]-y1;
            scalar_t z2 = buf[k*3 + 11] - z1;
            scalar_t d = x2 * x2 + y2 * y2 + z2 * z2;

            if (d < best) {
              best = d;
              best_i = k + k2 + 3;
            }
            }
          }
        } else {
          for (int k = 0; k < end_ka; k += 4) {
            {
              scalar_t x2 = buf[k * 3 + 0] - x1;
              scalar_t y2 = buf[k * 3 + 1] - y1;
              scalar_t z2 = buf[k * 3 + 2] - z1;
              scalar_t d = x2 * x2 + y2 * y2 + z2 * z2;

              if (k == 0 || d < best) {
                best = d;
                best_i = k + k2;
              }
            }

            {
              scalar_t x2 = buf[k * 3 + 3] - x1;
              scalar_t y2 = buf[k * 3 + 4] - y1;
              scalar_t z2 = buf[k * 3 + 5] - z1;
              scalar_t d = x2 * x2 + y2 * y2 + z2 * z2;

              if (d < best) {
                best = d;
                best_i = k + k2 + 1;
              }
            }

            {
              scalar_t x2 = buf[k * 3 + 6] - x1;
              scalar_t y2 = buf[k * 3 + 7] - y1;
              scalar_t z2 = buf[k * 3 + 8] - z1;
              scalar_t d= x2 * x2 + y2 * y2 + z2 * z2;

              if (d < best) {
                best = d;
                best_i = k + k2 + 2;
              }
            }

            {
              scalar_t x2 = buf[k * 3 + 9] - x1;
              scalar_t y2 = buf[k * 3 + 10] - y1;
              scalar_t z2 = buf[k * 3 + 11] - z1;
              scalar_t d = x2 * x2 + y2 * y2 + z2 * z2;

              if (d < best) {
                best = d;
                best_i = k + k2 + 3;
              }
            }
          }
        }
        for (int k = end_ka; k < end_k; k++) {
          scalar_t x2 = buf[k * 3 + 0] - x1;
          scalar_t y2 = buf[k * 3 + 1] - y1;
          scalar_t z2 = buf[k * 3 + 2] - z1;
          scalar_t d = x2 * x2 + y2 * y2 + z2 * z2;

          if (k == 0 || d < best) {
            best = d;
            best_i = k+k2;
          }
        }

        if (k2 == 0 || result[(i * n + j)] > best) {
          result[(i * n + j)] = best;
          result_i[(i * n + j)] = best_i;
        }
      }
      __syncthreads();
    }
  }
}

template<typename scalar_t>
__global__ void sided_distance_forward_cuda_kernel_2d(
    int b, int n, const scalar_t * xyz,
    int m, const scalar_t * xyz2,
    scalar_t * result, int64_t * result_i) {
  const int batch=512;
  constexpr int dim = 2;
  __shared__ scalar_t buf[batch*dim];

  for (int i = blockIdx.x; i<b; i += gridDim.x){
    for (int k2 = 0; k2 < m; k2 += batch) {

      int end_k =  min(m, k2 + batch) - k2;

      for (int j = threadIdx.x; j < end_k * dim; j += blockDim.x) {
        buf[j]=xyz2[(i*m+k2)*dim+j];
      }

      __syncthreads();

      for (int j = threadIdx.x + blockIdx.y * blockDim.x; j < n; j += blockDim.x * gridDim.y) {
        scalar_t x1 = xyz[(i * n + j) * dim + 0];
        scalar_t y1 = xyz[(i * n + j) * dim + 1];

        int64_t best_i = 0;
        scalar_t best = 0;
        int end_ka = end_k - (end_k & dim);

        best = 10000.0;
        for (int k = 0; k < end_k; k ++) {
            scalar_t x2 = buf[k * dim + 0] - x1;
            scalar_t y2 = buf[k * dim + 1] - y1;
            scalar_t d = x2 * x2 + y2 * y2;

            if (d < best) {
                best = d;
                best_i = k + k2;
            }
        }

        if (k2 == 0 || result[(i * n + j)] > best) {
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
        cudaMallocManaged(&res_, src_size_ * tar_size_ * sizeof(float));
        cudaMallocManaged(&src_, src_size_ * 2 * sizeof(float));
        cudaMallocManaged(&tar_, tar_size_ * 2 * sizeof(float));
    }

    std::vector<PointT> SearchGpu::kdsearch_gpu(const std::vector<PointT>& src, const std::vector<PointT>& tar)
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


        dim3 threads = dim3(256);
        dim3 blocks = dim3(src.size() / threads.x);
        kernel<<<blocks, threads>>>(src_, src_size_, tar_, tar_size_, res_);
        cudaDeviceSynchronize();

        std::vector<PointT> result(src.size(), {0.0, 0.0});
        for (int i = 0; i < src.size(); i++)
        {
            int best_idx = 0;
            float min_dist = res_[0];
            for (int j = 0; j < tar_size_; j++)
            {
                auto dist = res_[i * tar_size_ + j];
                if (dist < min_dist)
                {
                    min_dist = dist;
                    best_idx = j;
                }
            }
            result[i] = tar[best_idx];
        }
        return result;
    }

    void SearchGpu::release()
    {
        cudaFree(src_);
        cudaFree(tar_);
        cudaFree(res_);
    }


    void SearchGpu1::set_param(int src_size, int tar_size)
    {
        src_size_ = src_size;
        tar_size_ = tar_size;
        cudaMallocManaged(&dist_, src_size_ * sizeof(float));
        cudaMallocManaged(&idx_, src_size_ * sizeof(int64_t));
        cudaMallocManaged(&src_, src_size_ * 3 * sizeof(float));
        cudaMallocManaged(&tar_, tar_size_ * 3 * sizeof(float));
    }

    std::vector<PointT> SearchGpu1::kdsearch_gpu(const std::vector<PointT>& src, const std::vector<PointT>& tar)
    {
        for (int i = 0; i < src.size(); i++)
        {
            src_[i * 3] = src[i][0];
            src_[i * 3 + 1] = src[i][1];
            src_[i * 3 + 2] = 0.0;
        }

        for (int i = 0; i < tar.size(); i++)
        {
            tar_[i * 3] = tar[i][0];
            tar_[i * 3 + 1] = tar[i][1];
            tar_[i * 3 + 2] = 0.0;
        }


        dim3 threads = dim3(256);
        dim3 blocks = dim3(src.size() / threads.x);
        int b = 1;
        // sided_distance_forward_cuda_kernel<float><<<blocks, threads>>>(b, src_size_, src_, tar_size_, tar_, dist_, idx_);
        sided_distance_forward_cuda_kernel<float><<<dim3(32, 16, 1), 512, 0>>>(b, src_size_, src_, tar_size_, tar_, dist_, idx_);
        cudaDeviceSynchronize();

        std::vector<PointT> result(src.size(), {0.0, 0.0});
        for (int i = 0; i < src_size_; i++)
        {
            result[i] = tar[idx_[i]];
        }
        return result;
    }

    void SearchGpu1::release()
    {
        cudaFree(src_);
        cudaFree(tar_);
        cudaFree(idx_);
        cudaFree(dist_);
    }

    void SearchGpu2::set_param(int src_size, int tar_size)
    {
        src_size_ = src_size;
        tar_size_ = tar_size;
        cudaMallocManaged(&dist_, src_size_ * sizeof(float));
        cudaMallocManaged(&idx_, src_size_ * sizeof(int64_t));
        cudaMallocManaged(&src_, src_size_ * 2 * sizeof(float));
        cudaMallocManaged(&tar_, tar_size_ * 2 * sizeof(float));
    }

    std::vector<PointT> SearchGpu2::kdsearch_gpu(const std::vector<PointT>& src, const std::vector<PointT>& tar)
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

        std::vector<PointT> result(src.size(), {0.0, 0.0});
        for (int i = 0; i < src_size_; i++)
        {
            result[i] = tar[idx_[i]];
        }
        return result;
    }

    void SearchGpu2::release()
    {
        cudaFree(src_);
        cudaFree(tar_);
        cudaFree(idx_);
        cudaFree(dist_);
    }