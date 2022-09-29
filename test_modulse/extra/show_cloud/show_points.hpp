#pragma once
#include "open3d/Open3D.h"
#include <vector>

namespace show
{
    template<typename PointT>
    void show_points(const std::vector<PointT>& points)
    {
        std::shared_ptr<open3d::geometry::PointCloud> cloud 
            = std::make_shared<open3d::geometry::PointCloud>();
        for (const auto& point : points)
            cloud->points_.emplace_back(point.x, point.y, 0.0);
        open3d::visualization::DrawGeometries({cloud});
    }

    template<typename PointT>
    void show_points_normals(const std::vector<PointT>& points,
        const std::vector<PointT>& norms)
    {
        std::shared_ptr<open3d::geometry::PointCloud> cloud 
            = std::make_shared<open3d::geometry::PointCloud>();
        for (const auto& point : points)
            cloud->points_.emplace_back(point.x, point.y, 0.0);
        for (const auto& norm : norms)
            cloud->normals_.emplace_back(norm.x, norm.y, 0.0);
        open3d::visualization::DrawGeometries({cloud});
    }
}



