#include "ransac/ransac.hpp"
#include "ransac/ransac_shape.hpp"
#include "mylog/mylog.hpp"

namespace ransac
{
	double compute_error(const Circle2d &circle,
						 const std::vector<PointT> &points, std::vector<double> &dists)
	{
		double dist_sum = 0.0;
		int i = 0;
		int count = 0;
		for (const auto &point : points)
		{
			double dist = circle.compute_dist(point);
			dists[i] = dist;
			if (dist > 4.0)
				continue;
			dist_sum += dist;
			count++;
		}
		return dist_sum / (double)count;
	}

	double compute_error(const Ellipses2d &circle,
						 const std::vector<PointT> &points, std::vector<double> &dists)
	{
		double dist_sum = 0.0;
		int i = 0;
		int count = 0;
		for (const auto &point : points)
		{
			double dist = circle.compute_dist(point);
			dists[i] = dist;
			if (dist > 4.0)
				continue;
			dist_sum += dist;
			count++;
		}
		return dist_sum / (double)count;
	}

	size_t get_rand_index(const size_t &range)
	{
		return rand() % range;
	}

	std::vector<size_t> select_index_dist(size_t point_size, size_t select_size,
										  const std::vector<double> &dists)
	{
		select_size = std::min(select_size, point_size);

		std::unordered_set<size_t> st;
		while (st.size() < select_size)
		{
			size_t idx = get_rand_index(point_size);
			if (dists[idx] < 2.0)
				st.insert(idx);
		}
		std::vector<size_t> indexs(st.begin(), st.end());
		return indexs;
	}

	Circle2d fit_circle(const std::vector<PointT> &points, int iters, int select_points_num)
	{
		Circle2d circle;
		size_t point_size = points.size();
		std::vector<double> dists(point_size, 0);

		int best_count = 0;
		double best_error = std::numeric_limits<double>::max();
		for (int i = 0; i < iters; i++)
		{
			auto indexs = select_index_dist(point_size, select_points_num, dists);
			Circle2d circle_temp;
			circle_temp.set_param(points, indexs, false);
			double error = compute_error(circle_temp, points, dists);
			if (error < best_error)
			{
				best_error = error;
				circle = circle_temp;
			}
		}
		return circle;
	}

	Ellipses2d fit_ellipese(const std::vector<PointT> &points, int iters, int select_points_num)
	{
		Ellipses2d circle;
		size_t point_size = points.size();
		std::vector<double> dists(point_size, 0);

		int best_count = 0;
		double best_error = std::numeric_limits<double>::max();
		
		for (int i = 0; i < iters; i++)
		{
			auto indexs = select_index_dist(point_size, select_points_num, dists);
			Ellipses2d circle_temp;
			circle_temp.set_param(points, indexs, false);
			double error = compute_error(circle_temp, points, dists);
			if (error < best_error)
			{
				best_error = error;
				circle = circle_temp;
			}
		}

		LOG_OUT("===========best error ", best_error,  circle.height , circle.width ," ==================");
		return circle;
	}

}