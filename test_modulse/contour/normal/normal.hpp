#pragma once
#include <Eigen/Eigen>
#include "mylog/mylog.hpp"

namespace profile
{
    template<typename PointT>
    Eigen::Matrix2d compute_cov(const std::vector<int>& indexs, const std::vector<PointT>& points)
    {
        Eigen::Matrix2d covariance;
        Eigen::Matrix<double, 5, 1> cumulants;
        cumulants.setZero();
        for (const auto& idx : indexs) {
            const auto point = points[idx];
            cumulants(0) += point.x;
            cumulants(1) += point.y;
            cumulants(2) += point.x * point.x;
            cumulants(3) += point.x * point.y;
            cumulants(4) += point.y * point.y;
        }
        cumulants /= (double)indexs.size();
        covariance(0, 0) = cumulants(2) - cumulants(0) * cumulants(0);
        covariance(1, 1) = cumulants(4) - cumulants(1) * cumulants(1);
        covariance(0, 1) = cumulants(3) - cumulants(0) * cumulants(1);
        covariance(1, 0) = covariance(0, 1);
        return covariance;
    }

    template<typename PointT>
    PointT compute_normal(const Eigen::Matrix2d& cov)
    {
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> solver;
        solver.compute(cov, Eigen::ComputeEigenvectors);
        auto norm = solver.eigenvectors().col(0);
        return {(float)norm[0], (float)norm[1]};
    }

    template<typename PointT>
    std::vector<PointT> compute_normals(const std::vector<PointT>& points)
    {
        const int k = 4;
        std::vector<PointT> normals(points.size());
        for (int i = 1; i < points.size() - 2; i++)
        {   
            std::vector<int> indexs{i - 1, i, i + 1, i + 2};
            auto cov = compute_cov<PointT>(indexs, points);
            auto norm = compute_normal<PointT>(cov);
            normals[i] = norm;
        }
        normals[0] = normals[1];
        normals[normals.size() - 1] = normals[normals.size() - 3]; 
        normals[normals.size() - 2] = normals[normals.size() - 3];

        return normals; 
    }

    template<typename PointT>
    std::vector<float> compute_curvature(const std::vector<PointT>& points)
    {
        const int k = 4;
        std::vector<float> curvatures(points.size());
        for (int i = 1; i < points.size() - 2; i++)
        {   
            auto n1 = points[i - 1] - points[i];
            auto n2 = points[i + 1] - points[i];
            curvatures[i] = n1.dot(n2);
        }
        curvatures[0] = curvatures[1];
        curvatures[curvatures.size() - 1] = curvatures[curvatures.size() - 3]; 
        curvatures[curvatures.size() - 2] = curvatures[curvatures.size() - 3];

        return curvatures; 
    }
}

