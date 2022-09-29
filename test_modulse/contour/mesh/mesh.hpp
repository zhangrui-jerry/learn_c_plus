#pragma once
#include <vector>
#include <Eigen/Eigen>
#include "mylog/mylog.hpp"

template <typename PointT>
class Mesh
{
public:
    Mesh() : profile_count_{0}, cur_{0}, pre_{0} {}

    void add_points(const std::vector<PointT> &points, float z)
    {
        for (int i = 0; i < points.size(); i++)
        {
            const auto &point = points[i];
            points_.emplace_back(point.x, point.y, z);
        }

        // 第一帧不处理
        if (profile_count_ == 0)
        {
            pre_ = 0;
            cur_ = points_.size();
            profile_count_++;
            return;
        }

        int i = pre_;
        int j = cur_;
        int end_i = cur_;
        int end_j = points_.size();

        // 生成两个轮廓间的三角形
        make_triangles(i, j, end_i, end_j);
        
        cur_ = end_j;
        pre_ = end_i;

        profile_count_++;
    }

    std::vector<Eigen::Vector3i> triangles_;
    std::vector<Eigen::Vector3d> points_;

private:
    void make_triangles(int i, int j, int end_i, int end_j)
    {
        auto pi = points_[i];
        auto pj = points_[j];
        while (i < end_i - 1 && j < end_j - 1)
        {
            const auto &next_pi = points_[i + 1];
            const auto &next_pj = points_[j + 1];

            auto dist_i = (next_pj - pi).squaredNorm();
            auto dist_j = (next_pi - pj).squaredNorm();

            if (dist_i < dist_j)
            {
                triangles_.emplace_back(j, i, j + 1);
                pj = next_pj;
                j++;
            }
            else
            {
                triangles_.emplace_back(j, i, i + 1);
                pi = next_pi;
                i++;
            }
        }

        while (i < end_i - 1)
        {
            triangles_.emplace_back(j, i, i + 1);
            i++;
        }
        while (j < end_j - 1)
        {
            triangles_.emplace_back(j, i, j + 1);
            j++;
        }

        triangles_.emplace_back(j, i, pre_);
        triangles_.emplace_back(j, pre_, cur_);
    }
    int profile_count_;
    int pre_;
    int cur_;
};