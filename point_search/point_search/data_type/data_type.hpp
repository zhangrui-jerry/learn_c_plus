#pragma once
#include <vector>
#include <Eigen/Core>
#include "cuda_allocator.h"

using PointT = Eigen::Vector2d;
using CloudGPU = std::vector<PointT, CudaAllocator<PointT>>;
using Cloud = std::vector<PointT>;
using Indexs = std::vector<int, CudaAllocator<int>>;
using Dists = std::vector<double, CudaAllocator<double>>;