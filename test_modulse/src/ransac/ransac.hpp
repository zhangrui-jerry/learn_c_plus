#pragma once
#include <unordered_set>
#include <algorithm>
#include <Eigen/Eigenvalues>

#include "data_type/data_type.hpp"
#include "ransac/ransac_shape.hpp"

namespace ransac
{
	Circle2d fit_circle(const std::vector<PointT> &points, int iters = 200, int select_points_num = 30);
	Ellipses2d fit_ellipese(const std::vector<PointT> &points, int iters = 200, int select_points_num = 30);

	inline Circle2d fit_circle(const std::vector<cv::Point2f> &pts, int iters = 200, int select_points_num = 30)
	{
		std::vector<PointT> points(pts.size());
		for (int i = 0; i < pts.size(); i++)
			points[i] = {pts[i].x, pts[i].y};
		return fit_circle(points);
	}

	inline Ellipses2d fit_ellipese(const std::vector<cv::Point2f> &pts, int iters = 200, int select_points_num = 40)
	{
		std::vector<PointT> points(pts.size());
		for (int i = 0; i < pts.size(); i++)
			points[i] = {pts[i].x, pts[i].y};
		return fit_ellipese(points);
	}
}
