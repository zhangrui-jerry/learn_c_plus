
#include "TimeTest/TimeTest.hpp"
#include <data_type/data_type.hpp>
#include <kdtree_gpu.h>
#include <iostream>
#include <fstream>

std::vector<PointT> generate_data(const int& n)
{
    std::vector<PointT> data(n);
    for (int i = 0; i < n; i++)
        data[i] = {(rand() % 1000) / 1000.0, (rand() % 1000) / 1000.0};
    return data;
}

std::vector<PointT> read_data(const std::string& file_name)
{
    std::ifstream file(file_name);
    if (!file)
    {
        std::cout << "test file " + file_name + " not open!" << std::endl; 
        return {};
    }
    std::vector<PointT> data;
    double x{};
    double y{};
    while (file >> x >> y)
        data.emplace_back(x, y);
    return data;
}

std::vector<PointT> read_points_csv(const std::string& file_name)
{
    FILE *fp = fopen(file_name.c_str(), "r");
    if (fp == NULL)
    {
        std::cout << "file not open" << std::endl;
        return {};
    }
    char c;
    while (fscanf(fp, "%c", &c))
    {
        if (c == '\n')
            break;
    }
    std::vector<PointT> data;

    int laser_id{};
    float x{};
    float y{};
    while (fscanf(fp, "%d, %f, %f", &laser_id, &x, &y) != EOF)
        data.emplace_back(x, y);
    return data;
}

int search_violent(const std::vector<PointT>& tar, const PointT& e)
{
    double min_dist = 10000.0;
    int best_index{};
    for (int i = 0; i < tar.size(); i++)
    {
        const auto& point = tar[i];
        double dist = (point - e).norm();
        if (dist < min_dist)
        {
            min_dist = dist;
            best_index = i;
        }
    }
    return best_index;
}

std::vector<int> generate_res(const std::vector<PointT>& tar, const std::vector<PointT>& src)
{
    std::vector<int> res(src.size());
    for (int i = 0; i < src.size(); i++)
    {
        res[i] = search_violent(tar, src[i]);
    }
    return res;
}

void test_kdtree_case_0(const std::vector<PointT>& tar, 
    const std::vector<PointT>& src, 
    const std::vector<int>& res_true)
{
    SearchGpu sg;
    sg.set_param(src.size(), tar.size());
    START_TIME("GPU SEARCH");
    auto res = sg.kdsearch_gpu(src, tar);
    END_TIME("GPU SEARCH");

    int count = 0;
    for (int i = 0; i < src.size(); i++)
    {
        if (res[i] == res_true[i])
            count++;
    }
    // std::cout << "test_kdtree_case_1: " << (double)count / src.size() << std::endl;
    sg.release();
}

int main()
{
    INIT_TIME();
    // src = read_data("./test/profile/kdtree/data/rail_0222_all_1.pts");
    // tar = read_data("./test/profile/kdtree/data/rail_model_0222_all.pts");
    auto src = read_data("./test/data/rail_0222_all_2.txt");
    auto tar = read_data("./test/data/rail_model_0222_all.txt");
    auto res = generate_res(tar, src);
    for (int i = 0; i < 1000; i++)
        test_kdtree_case_0(tar, src, res);

    PRINTF_TIME();
    return 0;
}