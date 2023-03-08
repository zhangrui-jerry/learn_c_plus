#pragma once
#include <vector>
#include <Eigen/Core>
#include "cuda_allocator.h"

using PointT = Eigen::Vector2d;
using Cloud = std::vector<PointT, CudaAllocator<PointT>>;