#pragma once
#include "opencv2/core.hpp"

#include "mylog/mylog.hpp"
#include "data_convert/data_convert.hpp"
#include "area/profile_area.hpp"
#include "sort_profile/sort_profile.hpp"

#include <vector>

namespace contour
{
    void test_rail()
    {
        auto points = dataset::get_one_profile<cv::Point2f>();
        profile::ProfileSort<cv::Point2f> ps;
        auto points_sort = ps.process(points);
        auto area = profile::compute_profile_area(points_sort);
        std::cout<< area << std::endl;
    }

    void test()
    {
        test_rail();
    }
}
