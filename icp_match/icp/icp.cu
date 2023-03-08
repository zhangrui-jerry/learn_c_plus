#include <icp.hpp>
#include <data_type/data_type.hpp>

namespace icp
{
    template<typename T>
    void generate_cloud(std::shared_ptr<T> cloud, const int& size = 1000)
    {
        cloud->resize(size);
        auto randd = []{return static_cast<double>(rand() % 1024 / 1024.0);};
        for (int i = 0; i < size;  i++)
            (*cloud)[i] = {randd(), randd()};
    }

template<typename PointT, typename scalar_t>
__global__ void sided_distance_forward_cuda_kernel_2d(
    int b, int n, const PointT* xyz,
    int m, const PointT* xyz2,
    double* result, int64_t* result_i) 
{
//     const int batch=256;
//     __shared__ PointT buf[batch];

//   {
//     for (int k2 = 0; k2 < m; k2 += batch) {

//       int end_k =  min(m - k2, batch);
//       for (int j = threadIdx.x; j < end_k; j += blockDim.x) {
//         buf[j]=xyz2[k2+j];
//       }

//       __syncthreads();

//       for (int j = threadIdx.x + blockIdx.y * blockDim.x; j < n; j += blockDim.x * gridDim.y) {
//         scalar_t x1 = xyz[j][0];
//         scalar_t y1 = xyz[j][1];

//         int64_t best_i = 0;
//         scalar_t best = 0;
//         best = 10000.0;
//         for (int k = 0; k < end_k; k ++) {
//             scalar_t x2 = buf[k][0] - x1;
//             scalar_t y2 = buf[k][1] - y1;
//             scalar_t d = x2 * x2 + y2 * y2;

//             if (d < best) {
//                 best = d;
//                 best_i = k + k2;
//             }
//         }

//         if (k2 == 0 || result[j] > best) {
//           result[j] = best;
//           result_i[j] = best_i;
//         }
//       }
//       __syncthreads();
//     }
//   }
}
    int test()
    {
        std::shared_ptr<CloudGPU> src = std::make_shared<CloudGPU>();
        std::shared_ptr<CloudGPU> tar = std::make_shared<CloudGPU>();
        std::shared_ptr<Indexs> indexs = std::make_shared<Indexs>();
        std::shared_ptr<Dists> dists = std::make_shared<Dists>();

        generate_cloud(src);
        generate_cloud(tar);

        indexs->resize(src->size());
        dists->resize(src->size());
        int b = 1;
        sided_distance_forward_cuda_kernel_2d<PointT, double><<<dim3(32, 16, 1), 512, 0>>>
            (b, (int)src->size(), src->data(), (int)tar->size(), tar->data(), dists->data(), indexs->data());

        int count = 0; 
        for (int i = 0; i < src->size(); i++)
        {
            double min_dist = std::numeric_limits<double>::max();
            int best_index = 0;
            for (int j = 0; j < tar->size(); j++)
            {
                double dist = (src->at(i) - tar->at(j)).norm();
                if (dist < min_dist)
                {
                    min_dist = dist;
                    best_index = j;
                }
            }
            if (best_index == indexs->at(i))
                count++;
            // printf("%d %d\n", best_index, indexs->at(i));
        }
        printf("count %d\n", count);
        return 0;
    }
}
