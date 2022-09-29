#pragma once
#include <queue>
#include <algorithm>
#include <tuple>
#include <vector>

namespace profile
{
    template <typename PointT>
    class ProfileSort
    {
    public:
        ProfileSort() : max_x_{0}, max_y_{0}, min_x_{0}, min_y_{0}, cell_size_{0.5}, width_{0}, height_{0} {}
        std::vector<PointT> process(const std::vector<PointT> &points, float cell_size = 1.0)
        {
            if (points.empty())
                return std::vector<PointT>{};

            cell_size_ = cell_size;

            get_bound(points);
            create_grid(points);
            search();
            return points_res_;
        }

    private:
        float convert_x(float x)
        {
            return (int)((x - min_x_) / cell_size_);
        }

        float convert_y(float y)
        {
            return (int)((y - min_y_) / cell_size_);
        }

        void get_bound(const std::vector<PointT> &points)
        {
            max_x_ = points[0].x;
            max_y_ = points[0].y;
            min_x_ = points[0].x;
            min_y_ = points[0].y;

            for (const auto &point : points)
            {
                max_x_ = std::max(max_x_, point.x);
                max_y_ = std::max(max_y_, point.y);
                min_x_ = std::min(min_x_, point.x);
                min_y_ = std::min(min_y_, point.y);
            }

            width_  = convert_x(max_x_) + 1;
            height_ = convert_y(max_y_) + 1;
        }

        void create_grid(const std::vector<PointT> &points)
        {
            std::vector<int> cnts(points.size(), 2);
            points_grid_ = points;
            grid_.resize(width_ * height_, -1);

            // 将点放入格子中
            for (int i = 0; i < points_grid_.size(); i++)
            {
                int x = convert_x(points_grid_[i].x);
                int y = convert_y(points_grid_[i].y);

                auto &temp = grid_[y * width_ + x];
                if (temp == -1) // 如果这个格子没点
                {
                    temp = i;
                }
                else // 如果这个有点，求平均
                {
                    const float cnt = (float)(cnts[temp]);
                    points_grid_[temp].x = points_grid_[temp].x * (cnt - 1) / cnt + points_grid_[i].x / cnt;
                    points_grid_[temp].y = points_grid_[temp].y * (cnt - 1) / cnt + points_grid_[i].y / cnt;
                    cnts[temp]++;
                }
                // temp = i;
            }
        }

        bool check(int x, int y)
        {
            return x >= 0 && x < width_ && y >= 0 && y < height_ && grid_[y * width_ + x] != -1;
        }

        void search_init()
        {
            int x0 = convert_x(points_grid_[0].x);
            int y0 = convert_y(points_grid_[0].y);

            int lx = x0;
            int rx = x0;
            int ly = y0;
            int ry = y0;

            // 对初始点建立十字划分，使得单方向增长
            while (true)
            {
                if (check(lx - 1, y0) || check(rx + 1, y0) || check(x0, ly - 1) || check(x0, ry + 1))
                {
                    if (check(lx - 1, y0))
                        lx--;
                    if (check(rx + 1, y0))
                        rx++;
                    if (check(x0, ly - 1))
                        ly--;
                    if (check(x0, ry + 1))
                        ry++;
                }
                else
                {
                    if (rx - lx < ry - ly)
                    {
                        for (int i = lx; i <= rx; i++)
                        {
                            auto &temp = grid_[y0 * width_ + i];
                            points_res_.push_back(points_grid_[temp]);
                            temp = -1;
                        }
                        que_.push({x0, y0 + 1});
                    }
                    else
                    {
                        for (int i = ly; i <= ry; i++)
                        {
                            auto &temp = grid_[i * width_ + x0];
                            points_res_.push_back(points_grid_[temp]);
                            temp = -1;
                        }
                        que_.push({x0 + 1, y0});
                    }
                    break;
                }
            }
        }

        void search()
        {
            search_init();
            
            // 广度遍历，轮廓生长
            while (!que_.empty())
            {
                auto [x, y] = que_.front();
                que_.pop();
                for (int i = -1; i <= 1; i++)
                {
                    for (int j = -1; j <= 1; j++)
                    {
                        int xx = x + i;
                        int yy = y + j;
                        if (xx >= 0 && xx < width_ && yy >= 0 && yy < height_)
                        {
                            auto &temp = grid_[yy * width_ + xx];
                            if (temp != -1)
                            {
                                que_.push({xx, yy});
                                points_res_.push_back(points_grid_[temp]);
                                temp = -1;
                            }
                        }
                    }
                }
            }
        }

        float max_x_;
        float max_y_;
        float min_x_;
        float min_y_;
        int width_;
        int height_;
        float cell_size_;
        std::vector<int> grid_;
        std::vector<PointT> points_grid_;
        std::queue<std::pair<int, int>> que_;
        std::vector<PointT> points_res_;
    };
}