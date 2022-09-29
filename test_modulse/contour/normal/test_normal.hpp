#pragma once
#include "normal.hpp"
#include "data_convert/data_convert.hpp"
#include "sort_profile/sort_profile.hpp"
#include "show_cloud/show_points.hpp"
#include <iostream>

namespace profile
{
    void test_normal()
    {
        auto data = dataset::get_one_profile<cv::Point2f>();
        // auto data = dataset::generate_circle<cv::Point2f>();
        profile::ProfileSort<cv::Point2f> ps;
        auto profile = ps.process(data);

        auto norm = profile::compute_normals(profile);
        auto curvature = profile::compute_curvature(data);

        for (auto& cur : curvature)
        {
            std::cout << cur << std::endl;
        }

        show::show_points_normals(profile, norm);
    }
}