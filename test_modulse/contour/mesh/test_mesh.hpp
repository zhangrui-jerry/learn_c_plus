#pragma once
#include "data_convert/data_convert.hpp"
#include "mesh.hpp"
#include "show_mesh.hpp"
#include "sort_profile/sort_profile.hpp"
#include "opencv2/core.hpp"
#include "show_cloud/show_points.hpp"
#include "mylog/mylog.hpp"

namespace contour
{
    std::vector<cv::Point2f> mean_filter(const std::vector<cv::Point2f> &points, int kernel_size = 7)
    {
        if (points.size() < kernel_size)
        {
            LOG_OUT("mean filter kernel size less than kernel size");
            return points;
        }
        std::vector<cv::Point2f> pts(points.size());
        for (int i = 0; i < points.size(); i++)
        {
            cv::Point2f point{0, 0};
            for (int j = i; j < i + kernel_size; j++)
            {
                int jj = j % points.size();
                point.x += points[jj].x;
                point.y += points[jj].y;
            }

            point.x /= (float)(kernel_size);
            point.y /= (float)(kernel_size);

            pts[i] = point;
        }

        return pts;
    }
    void test_mesh()
    {

        Mesh<cv::Point2f> mesh;
        for (float z = 0; z < 50; z += 0.2)
        {
            auto points = dataset::get_one_profile<cv::Point2f>();

            profile::ProfileSort<cv::Point2f> ps;
            auto points_sort = ps.process(points);

            // if (z < 0.1)
            //     show::show_points<cv::Point2f>(points_sort);

            auto points_mean = mean_filter(points_sort);
            if (z < 0.1)
                show::show_points<cv::Point2f>(points_mean);

            mesh.add_points(points_mean, z);
        }

        mesh::show_mesh(mesh);
    }
}
