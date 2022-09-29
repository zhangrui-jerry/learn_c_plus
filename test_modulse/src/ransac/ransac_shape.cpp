#include "ransac/ransac_shape.hpp"
#include "mylog/mylog.hpp"
#include "opencv2/opencv.hpp"

#define ELLIPESE_PARAM 100.0

namespace ransac
{
	void Circle2d::set_param(const std::vector<PointT> &points,
							 const std::vector<size_t> &indexs, bool use_radius)
	{
		const size_t size = indexs.size();
		Eigen::MatrixXd A;
		Eigen::MatrixXd b;
		Eigen::MatrixXd x;

		A = Eigen::MatrixXd::Random(size, 3);
		b = Eigen::MatrixXd::Random(size, 1);
		x = Eigen::MatrixXd::Random(3, 1);

		// 两种求解方式，一种是规定半径拟合圆，一种是无限制拟合圆
		if (use_radius)
		{
			size_t i = 0;
			for (const auto &index : indexs)
			{
				const auto &p = points[index];
				b(i, 0) = 100.0 - p.squaredNorm();
				A(i, 0) = p[0];
				A(i, 1) = p[1];
				A(i, 2) = 1.0;
				i++;
			}
			x = A.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV)
					.solve(b)
					.transpose();

			radius = 10.0;
			center = {-x(0) / 2.0, -x(1) / 2.0};
		}
		else
		{
			size_t i = 0;
			for (const auto &index : indexs)
			{
				const auto &point = points[index];
				b(i, 0) = -point.squaredNorm();
				A(i, 0) = point[1];
				A(i, 1) = point[0];
				A(i, 2) = 1.0;
				i++;
			}
			x = A.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV)
					.solve(b)
					.transpose();

			D = x(0);
			E = x(1);
			F = x(2);

			radius = std::sqrt(D * D + E * E - 4 * F) / 2;
			center = {-E / 2, -D / 2};
		}
	}

	double Circle2d::compute_dist(const PointT &p) const
	{
		return std::abs(radius - (p - center).norm());
	}

	double Ellipses2d::compute_dist(const PointT &p) const
	{
		double x = p[0];
		double z = p[1];
		auto res = std::abs(x * x * coe[0] + x * z * coe[1] + z * z * coe[2] + x * coe[3] + z * coe[4] + coe[5]);
		return res;
	}

	void Ellipses2d::set_param(const std::vector<PointT> &points,
							   const std::vector<size_t> &indexs, bool use_radius)
	{
		const size_t size = indexs.size();
		Eigen::MatrixXd A;
		Eigen::MatrixXd b;
		Eigen::MatrixXd X;

		A = Eigen::MatrixXd::Random(size, 5);
		b = Eigen::MatrixXd::Random(size, 1);
		X = Eigen::MatrixXd::Random(5, 1);

		for (int i = 0; i < indexs.size(); i++)
		{
			const auto &index = indexs[i];
			const auto &point = points[index];
			double x = point[0];
			double z = point[1];

			b(i, 0) = ELLIPESE_PARAM;

			A(i, 0) = x * x;
			A(i, 1) = x * z;
			A(i, 2) = z * z;
			A(i, 3) = x;
			A(i, 4) = z;
		}

		X = A.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV)
				.solve(b)
				.transpose();

		auto compute_param = [](const Eigen::MatrixXd &res)
		{
			double A = res(0);
			double B = res(1);
			double C = res(2);
			double D = res(3);
			double E = res(4);
			double F = -ELLIPESE_PARAM;

			double det = 4 * A * C - B * B;
			// LOG_OUT(det);
			if (std::abs(det) < 1e-9)
				return std::make_tuple((double)(0), (double)(0), (double)(0), (double)(0));
			double center_x = (B * E - 2 * C * D) / det;
			double center_y = (B * D - 2 * A * E) / det;

			double tb = A + C - std::sqrt(std::pow(A - C, 2) + std::pow(B, 2));
			double ta = A + C + std::sqrt(std::pow(A - C, 2) + std::pow(B, 2));
			if (std::abs(tb) < 1e-9)
				return std::make_tuple(center_x, center_y, (double)(0), (double)(0));

			double temp = 2 * (A * std::pow(center_x, 2) + C * std::pow(center_y, 2) + B * center_x * center_y - F);
			double w = temp / ta;
			double h = temp / tb;
			w = std::sqrt(w);
			h = std::sqrt(h);
			return std::make_tuple(center_x, center_y, w, h);
		};

		auto [center_x, center_y, w, h] = compute_param(X);
		width = w;
		height = h;
		center[0] = center_x;
		center[1] = center_y;

		coe = {X(0), X(1), X(2), X(3), X(4), -ELLIPESE_PARAM};
	}

	std::vector<double> get_param(const cv::RotatedRect &ellipse)
	{
		double x0 = ellipse.center.x;
		double y0 = ellipse.center.y;

		double aa = ellipse.size.width;
		double bb = ellipse.size.height;
		double phi_b_rad = ellipse.angle * 180 / 3.1415926;

		double a = aa / 2;
		double b = bb / 2;

		double ax = -std::sin(phi_b_rad);
		double ay = std::cos(phi_b_rad);

		double a2 = a * a;
		double b2 = b * b;

		//  Ax ^ 2 + Bxy + Cy ^ 2 + Dx + Ey + F
		if (a2 < 0 || b2 < 0)
			return {1.0, 0, 1.0, 0, 0, -1e-6};

		double A = ax * ax / a2 + ay * ay / b2;
		double B = 2 * ax * ay / a2 - 2 * ax * ay / b2;
		double C = ay * ay / a2 + ax * ax / b2;
		double D = (-2 * ax * ay * y0 - 2 * ax * ax * x0) / a2 + (2 * ax * ay * y0 - 2 * ay * ay * x0) / b2;
		double E = (-2 * ax * ay * x0 - 2 * ay * ay * y0) / a2 + (2 * ax * ay * x0 - 2 * ax * ax * y0) / b2;
		double F = (2 * ax * ay * x0 * y0 + ax * ax * x0 * x0 + ay * ay * y0 * y0) / a2 +
				   (-2 * ax * ay * x0 * y0 + ay * ay * x0 * x0 + ax * ax * y0 * y0) / b2 - 1;

		std::vector<double> res{A, B, C, D, E, F};
		double rate = ELLIPESE_PARAM / F;
		for (auto &val : res)
			val *= rate;

		return res;
	}

	void Ellipses2d::set_param_cv(const std::vector<PointT> &points,
					  const std::vector<size_t> &indexs, bool use_radius)
	{
		std::vector<cv::Point2f> pts(points.size());
		for (int i = 0; i < points.size(); i++)
			pts[i] = {(float)points[i][0], (float)points[i][1]};
		auto ellipse = cv::fitEllipse(pts);
		coe = get_param(ellipse);
		center = {ellipse.center.x, ellipse.center.y};

		width = ellipse.size.width;
		height = ellipse.size.height;
		angle = ellipse.angle;
	}
}