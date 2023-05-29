#include <icp.hpp>
#include <data_type/data_type.hpp>
#include <TimeTest/TimeTest.hpp>
#include <kdtree_gpu.h>

namespace icp
{
    template <typename T>
    void generate_cloud(std::shared_ptr<T> cloud, const int &size = 4096 * 8)
    {
        cloud->resize(size);
        auto randd = []
        { return static_cast<double>(rand() % 1024 / 1024.0); };
        for (int i = 0; i < size; i++)
            (*cloud)[i] = {randd(), randd()};
    }

    template <typename Func>
    __global__ void parallel_for(int n, Func func)
    {
        for (int i = threadIdx.x + blockIdx.y * blockDim.x; i < n; i += blockDim.x * gridDim.y)
            func(i);
    }

    template <typename PointType>
    __global__ void search_nn(const PointType *src, const PointType *tar,
                              int src_size, int tar_size, int *indexs, double *dists)
    {
        const int batch = 256;
        __shared__ PointType buf[batch];

        for (int k = 0; k < tar_size; k += batch)
        {
            int batch_size = min(tar_size - k, batch);
            for (int i = threadIdx.x; i < batch_size; i += blockDim.x)
            {
                buf[i] = tar[k + i];
            }

            __syncthreads();

            for (int i = threadIdx.x + blockIdx.y * blockDim.x; i < src_size; i += blockDim.x * gridDim.y)
            {
                const auto &q = src[i];
                int best_index = 0;
                double min_dist = 10000.0;
                for (int j = 0; j < batch_size; j++)
                {
                    const auto &p = buf[j];
                    double dist = (p - q).squaredNorm();
                    if (dist < min_dist)
                    {
                        min_dist = dist;
                        best_index = j + k;
                    }
                }

                if (k == 0 || min_dist < dists[i])
                {
                    dists[i] = min_dist;
                    indexs[i] = best_index;
                }
            }

            __syncthreads();
        }
    }

    int test()
    {
        std::shared_ptr<CloudGPU> src = std::make_shared<CloudGPU>();
        std::shared_ptr<CloudGPU> tar = std::make_shared<CloudGPU>();
        std::shared_ptr<Cloud> src_cpu = std::make_shared<Cloud>();
        std::shared_ptr<Cloud> tar_cpu = std::make_shared<Cloud>();
        std::shared_ptr<Indexs> indexs = std::make_shared<Indexs>();
        std::shared_ptr<Dists> dists = std::make_shared<Dists>();

        INIT_TIME();
        START_TIME("generate cloud");
        generate_cloud(src);
        generate_cloud(tar);
        END_TIME("generate cloud");

        START_TIME("generate cpu cloud");
        generate_cloud(src_cpu);
        generate_cloud(tar_cpu);
        END_TIME("generate cpu cloud");

        START_TIME("create sg");
        SearchGpu sg;
        sg.set_param(src_cpu->size(), tar_cpu->size());
        END_TIME("create sg");

        START_TIME("SG SEARCH");
        auto result = sg.kdsearch_gpu(*(src_cpu), *(tar_cpu));
        END_TIME("SG SEARCH");

        indexs->resize(src->size());
        dists->resize(src->size());
        int b = 1;
        START_TIME("gpu");
        // sided_distance_forward_cuda_kernel_2d<<<dim3(32, 16, 1), 512, 0>>>
        //     ((int)src->size(), src->data(), (int)tar->size(), tar->data(), dists->data(), indexs->data());
        search_nn<<<dim3(32, 16, 1), 512, 0>>>(src->data(), tar->data(), (int)src->size(),
                                               (int)tar->size(), indexs->data(), dists->data());

        cudaDeviceSynchronize();
        END_TIME("gpu");

        START_TIME("cpu");
        int count = 0;
        int count1 = 0;
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
            if ((result[i] - tar->at(best_index)).norm() < 0.01)
                count1++;
        }
        END_TIME("cpu");
        PRINTF_TIME();
        printf("count %d %d\n", count, count1);
        return 0;
    }
}
