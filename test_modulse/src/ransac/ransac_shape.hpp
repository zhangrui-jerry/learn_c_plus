#pragma once
#include "data_type/data_type.hpp"

namespace ransac
{
	class Circle2d
	{
	public:
		void set_param(const std::vector<PointT> &points,
					   const std::vector<size_t> &indexs, bool use_radius = false);

		double compute_dist(const PointT &p) const;

		double radius;
		PointT center;

	private:
		double D;
		double E;
		double F;
	};

	class Ellipses2d
	{
	public:
		void set_param(const std::vector<PointT> &points,
					   const std::vector<size_t> &indexs, bool use_radius = false);
		void set_param_cv(const std::vector<PointT> &points,
						  const std::vector<size_t> &indexs, bool use_radius = false);

		double compute_dist(const PointT &p) const;
		double width;
		double height;
		PointT center;
		double angle;

	private:
		std::vector<double> coe;
	};
}