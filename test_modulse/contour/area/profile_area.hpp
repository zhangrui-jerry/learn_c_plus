#pragma once
#include <vector>

namespace profile
{
    template <typename PointT>
    double compute_profile_area(const std::vector<PointT> &points)
    {
        double area = 0;
        PointT point_pre = points.back();
        for (int i = 0; i < points.size(); i++)
        {
            const auto &point = points[i];
            area += point_pre.x * point.y - point_pre.y * point.x;
            point_pre = point;
        }

        area *= 0.5;
        return std::abs(area);
    }
}