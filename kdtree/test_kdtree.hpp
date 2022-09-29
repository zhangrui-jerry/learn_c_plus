#pragma once
#include "kdtree.hpp"
#include "kdsearch.hpp"
#include "nanokdtree.hpp"
#include <vector>
#include <algorithm>
#include "TimeTest/TimeTest.hpp"

using namespace profile;


void test_kdtree()
{
#if 0
    std::vector<float> data(20);
    auto rand30 = [](){return static_cast<float>(rand() % 1024 / 1024.0);};
    std::generate(data.begin(), data.end(), rand30);

    for (const auto& d : data)
        std::cout << d << " ";
    std::cout << std::endl;

    Kdtree<float> kdtree;
    kdtree.set_data(data);
    kdtree.build_tree();
    kdtree.print();

    std::cout<< kdtree.search_nn(0.6);
#endif
}


std::vector<PointT> generate_data(int size)
{
    std::vector<PointT> data(size);
    auto rand30 = [](){return static_cast<float>(rand() % 102400 / 10240);};
    auto rand30T = [&rand30](){return PointT{rand30(), rand30()};};
    std::generate(data.begin(), data.end(), rand30T);
    return data;
}

void test_kdsearch()
{
    INIT_TIME();

    
    constexpr int n = 4000 * 100;
    auto data = generate_data(n / 100);
    Kdtree kdtree;
    kdtree.set_data(data);
    START_TIME("build nn tree");
    kdtree.build_tree();
    END_TIME("build nn tree");

    auto test_data = generate_data(n);

    std::vector<PointT> res0(n);
    START_TIME("violent search");
    for (int i = 0; i < n; i++)
    {
        auto& d = test_data[i];
        res0[i] = kdtree.search_violent(d);
    }
    END_TIME("violent search");

    std::vector<PointT> res1(n);
    START_TIME("nn search");
    for (int i = 0; i < n; i++)
    {
        auto& d = test_data[i];
        res1[i] = kdtree.search_nn(d).data;
    }
    END_TIME("nn search");

    KdTree<PointT, 2> nanokd;
    START_TIME("build nano tree");
    nanokd.set_cloud(data);
    END_TIME("build nano tree");

    std::vector<size_t> indexs(1);
    std::vector<double> dists(1);

    std::vector<PointT> res2(n);

    START_TIME("nano search");
    for (int i = 0; i < n; i++)
    {
        auto& d = test_data[i];
        nanokd.knnSearch(d, 1, indexs, dists);
        res2[i] = data[indexs[0]];
    }
    END_TIME("nano search");

    auto test_res = [&res0, &n](std::vector<PointT>& res)
    {
        int cnt = 0;
        for (int i = 0; i < n; i++)
        {
            auto dist = (res[i] - res0[i]).norm();
            if (dist > 0.01)
                cnt++;
        }
        return (double)(n - cnt) / (double)n;
    };

    std::cout << test_res(res1) << std::endl;
    std::cout << test_res(res2) << std::endl;

    PRINTF_TIME();

}