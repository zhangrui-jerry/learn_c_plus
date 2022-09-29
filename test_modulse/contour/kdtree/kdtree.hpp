#pragma once
#if 0
#include <vector>
#include <memory>
#include <algorithm>
#include <numeric>
#include <iostream>



template<typename T>
struct Node
{
    Node() 
        : left{nullptr}, right{nullptr}{}

    int l;
    int r;
    T split;
    T low;
    T high;
    std::shared_ptr<Node> left;
    std::shared_ptr<Node> right;

};

template<typename T>
struct Result
{
    T dist;
    T data;    
};
template<typename T>
class Kdtree
{
public:

    using Data = std::vector<T>;

    Kdtree() : data_{std::make_shared<Data>()}, head_{nullptr}{}

    void set_data(const Data& data)
    {
        *data_ = data;
    }

    void build_tree()
    {
        head_ = std::make_shared<Node<T>>();
        head_->l = 0;
        head_->r = data_->size() - 1;
        divide_tree(head_);
    }

    void divide_tree(std::shared_ptr<Node<T>> node)
    {
        auto left = node->l;
        auto right = node->r;

        if (right - left < 1)
            return;

        auto split = get_split(left, right);
        node->split = split;
        int mid = midlle_split(split, left, right);
        node->left = std::make_shared<Node<T>>();
        node->left->l = left;
        node->left->r = mid;

        node->right = std::make_shared<Node<T>>();
        node->right->l = mid + 1;
        node->right->r = right;
        divide_tree(node->left);
        divide_tree(node->right);
    }

    T get_split(int left, int right)
    {
        T sum = std::accumulate(data_->begin() + left, data_->begin() + right + 1, 0.0);
        return sum / static_cast<T>(right - left + 1);
    }

    int midlle_split(T split, int left, int right)
    {
        int i = left;
        int j = right;
        auto& data = *data_;
        while (i <= j)
        {
            while (i <= j && data[i] <= split)
                i++;
            while (i <= j && data[j] > split)
                j--;
            if (i < j)
                std::swap(data[i], data[j]);
        }
        return j;
    }

    T compute_dist(const T& a, const T& b)
    {
        return std::abs(a - b);
    }

    T search_nn(const T& e)
    {
        Result<T> res;
        res.dist = 10000.0;
        search_nn(e, res, head_);
        return res.data;
    }

    void search_nn(const T& e, Result<T>& res, std::shared_ptr<Node<T>> node)
    {
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
                }
            }
            return;
        }
        if (e <= node->split)
            search_nn(e, res, node->left);
        else
            search_nn(e, res, node->right);
    }

    void print()
    {
        for (const auto& d : *data_)
            std::cout << d << " ";
        std::cout << std::endl;
    }

    std::shared_ptr<Data> data_;  
    std::shared_ptr<Node<T>> head_;
};
#endif