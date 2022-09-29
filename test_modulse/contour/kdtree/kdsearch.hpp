#pragma once
#include <vector>
#include <memory>
#include <algorithm>
#include <numeric>
#include <iostream>
#include "Eigen/Core"


namespace profile
{
    using PointT = Eigen::Vector2d;
    using Box = std::array<PointT, 2>;
    struct Node
    {
        Node()
            : left{ nullptr }, right{ nullptr }, dim{ 0 }{}

        int l;
        int r;
        int dim;
        double split;
        Box box;
        std::shared_ptr<Node> left;
        std::shared_ptr<Node> right;

    };

    struct SearchResult
    {
        double dist;
        PointT data;
        int index;
    };

    class Kdtree
    {
    public:
        using Data = std::vector<PointT>;

        Kdtree() : data_{ std::make_shared<Data>() },
            indexs_{ std::make_shared<std::vector<int>>() },
            head_{ nullptr }{}

        void set_data(const Data& data)
        {
            *data_ = data;
            indexs_->resize(data_->size());
            for (int i = 0; i < indexs_->size(); i++)
                (*indexs_)[i] = i;
        }

        void build_tree()
        {
            head_ = std::make_shared<Node>();
            head_->l = 0;
            head_->r = data_->size() - 1;
            divide_tree(head_);
        }

        int get_next_dim(std::shared_ptr<Node> node)
        {
            if (node->box[1][0] > node->box[1][1])
                return 0;
            return 1;
        }

        void divide_tree(std::shared_ptr<Node> node)
        {
            auto left = node->l;
            auto right = node->r;

            get_split(left, right, node->dim, node->split, node->box);

            if (right - left < 1 ||
                std::abs((*data_)[left][node->dim] - (*data_)[right][node->dim]) < 0.1)
                return;

            int mid = midlle_split(node->split, left, right, node->dim);
            node->left = std::make_shared<Node>();
            node->left->l = left;
            node->left->r = mid;
            node->left->dim = get_next_dim(node);

            node->right = std::make_shared<Node>();
            node->right->l = mid + 1;
            node->right->r = right;
            node->right->dim = get_next_dim(node);

            divide_tree(node->left);
            divide_tree(node->right);
        }

        void get_split(int left, int right, int& dim, double& split, Box& box)
        {
            if (data_->empty())
                return;

            auto x = (*data_)[0][0];
            auto y = (*data_)[0][1];
            double max_x = x;
            double max_y = y;
            double min_x = x;
            double min_y = y;

            PointT sum{ 0, 0 };
            for (int i = left; i <= right; i++)
            {
                const auto& var = (*data_)[i];
                sum += var;

                x = var[0];
                y = var[1];

                max_x = std::max(x, max_x);
                max_y = std::max(y, max_y);
                min_x = std::min(x, min_x);
                min_y = std::min(y, min_y);
            }
            box[0][0] = (max_x + min_x) / 2.0;
            box[0][1] = (max_y + min_y) / 2.0;
            box[1][0] = (max_x - min_x) / 2.0;
            box[1][1] = (max_y - min_y) / 2.0;

            if (box[1][0] > box[1][1])
                dim = 0;
            else
                dim = 1;
            split = (sum / static_cast<double>(right - left + 1))[dim];
        }

        int midlle_split(double split, int left, int right, int dim)
        {
            int i = left;
            int j = right;
            auto& data = *data_;
            while (i <= j)
            {
                while (i <= j && data[i][dim] <= split)
                    i++;
                while (i <= j && data[j][dim] > split)
                    j--;
                if (i < j)
                {
                    std::swap(data[i], data[j]);
                    std::swap((*indexs_)[i], (*indexs_)[j]);
                }
            }
            return j;
        }

        inline double compute_dist(const PointT& a, const PointT& b)
        {
            return (a - b).norm();
        }

        SearchResult search_nn(const PointT& e)
        {
            SearchResult res;
            res.dist = 10000.0;
            search_nn(e, res, head_);
            return res;
        }

        void search_nn(const PointT& e, SearchResult& res, std::shared_ptr<Node> node)
        {
            //auto dist_x = std::abs(e[0] - node->box[0][0]) - node->box[1][0];
            //auto dist_y = std::abs(e[1] - node->box[0][1]) - node->box[1][1];
            //if (dist_x > res.dist || dist_y > res.dist)
            //    return;
            if (node->left == nullptr && node->right == nullptr)
            {
                for (int i = node->l; i <= node->r; i++)
                {
                    const auto& var = (*data_)[i];
                    auto dist = compute_dist(var, e);
                    if (dist < res.dist)
                    {
                        res.dist = dist;
                        res.data = var;
                        res.index = (*indexs_)[i];
                    }
                }
                return;
            }

            auto dist = e[node->dim] - node->split;
            if (dist < 0)
            {
                search_nn(e, res, node->left);
                if (res.dist < std::abs(dist))
                    return;
                search_nn(e, res, node->right);
            }
            else
            {
                search_nn(e, res, node->right);
                if (res.dist < std::abs(dist))
                    return;
                search_nn(e, res, node->left);
            }
        }

        PointT search_violent(const PointT& e)
        {
            double min_dist = 10000.0;
            PointT best_point{};
            for (const auto& point : *data_)
            {
                double dist = (point - e).norm();
                if (dist < min_dist)
                {
                    min_dist = dist;
                    best_point = point;
                }
            }
            return best_point;
        }
        std::shared_ptr<Data> data_;
        std::shared_ptr<std::vector<int>> indexs_;
        std::shared_ptr<Node> head_;
    };
}
