#pragma once
#include <vector>
#include <limits>
#include <memory>
#include "nanoflann/nanoflann.hpp"
#include "Eigen/Core"


/// <summary>
/// 支持2D点和3D点搜索
/// </summary>
/// <typeparam name="T">点类型</typeparam>
template<typename T, int DIM>
class KdTree
{
public:
	KdTree() {};
	KdTree(const std::vector<T>& points)
	{
		set_cloud(points);
	}

	void set_cloud(const std::vector<T>& points)
	{
		cloud_.points_ = points;
		index_.reset(new my_kd_tree_t(DIM, cloud_, nanoflann::KDTreeSingleIndexAdaptorParams(10)));
		index_->buildIndex();
	}

	int knnSearch(const T& point, int k, std::vector<size_t>& indexs, 
		std::vector<double>& dists) const
	{
		double temp_point[DIM];
		for (int i = 0; i < DIM; i++)
			temp_point[i] = point[i];
		int num_results = index_->knnSearch(temp_point, k, &indexs[0], &dists[0]);
		return num_results;
	}

	int radius_search(const T& point, double radius, 
		std::vector<size_t>& indexs, std::vector<double>& dists) const
	{
		std::vector<std::pair<size_t, double>> index_dist;
		double temp_point[DIM];
		for (int i = 0; i < DIM; i++)
			temp_point[i] = point[i];
		int num_results = index_->radiusSearch(temp_point, radius, index_dist, nanoflann::SearchParams());
		indexs.resize(num_results);
		dists.resize(num_results);
		for (int i = 0; i < num_results; i++)
		{
			auto& res = index_dist[i];
			indexs[i] = res.first;
			dists[i] = res.second;
		}
		return num_results;
	}
private:
	struct PointCloud
	{
		std::vector<T> points_;

		inline size_t kdtree_get_point_count() const { return points_.size(); }

		inline double kdtree_get_pt(const size_t idx, const size_t dim) const
		{
			return points_[idx][dim];
		}

		template <class BBOX>
		bool kdtree_get_bbox(BBOX&) const { return false; }
	};
	typedef nanoflann::KDTreeSingleIndexAdaptor<nanoflann::L2_Simple_Adaptor<double, PointCloud>, PointCloud, DIM> my_kd_tree_t;
	std::shared_ptr<my_kd_tree_t> index_;
	PointCloud cloud_;
};