#pragma once
#include "Eigen/Core"
#include <vector>

using PointT = Eigen::Vector2d;

class SearchGpu
{
public:
    void set_param(int src_size, int tar_size);
    std::vector<int> kdsearch_gpu(const std::vector<PointT>& src, const std::vector<PointT>& tar);
    void release();
private:
    float *src_;
    float *tar_;
    float *dist_;
    int64_t *idx_;
    int src_size_;
    int tar_size_;
};