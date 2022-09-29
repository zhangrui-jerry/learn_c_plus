#pragma once
#include <Eigen/Eigen>
#include <Eigen/core>
#include "opencv2/core.hpp"
#include <vector>
#include <array>
#include "dataset/dataset.hpp"
#include "dataset/dataset4.hpp"
#include "mylog/mylog.hpp"

#define PROFILE_X_STEP -0.2
#define PROFILE_Z_MIN_CODE -900

namespace dataset
{
    void change_trans(std::array<Eigen::Matrix3d, 4> &trans)
    {
        Eigen::Matrix3d trans_change[4];
        trans_change[0] << 1.0000000000000000, 0.0000000000000000, -0.2000000000000000,
            0.0000000000000000, 1.0000000000000000, 0.2000000000000000,
            0.0000000000000000, 0.0000000000000000, 1.0000000000000000;
        trans_change[1] << 0.9999984769132877, 0.0017453283658983, -0.3000000000000000,
            -0.0017453283658983, 0.9999984769132877, 0.0000000000000000,
            0.0000000000000000, 0.0000000000000000, 1.0000000000000000;
        trans_change[2] << 0.9999939076577904, 0.0034906514152237, 0.0000000000000000,
            -0.0034906514152237, 0.9999939076577904, -0.2500000000000000,
            0.0000000000000000, 0.0000000000000000, 1.0000000000000000;
        trans_change[3] << 1.0000000000000000, 0.0000000000000000, 0.0000000000000000,
            0.0000000000000000, 1.0000000000000000, 0.0000000000000000,
            0.0000000000000000, 0.0000000000000000, 1.0000000000000000;

        Eigen::Matrix3d trans_match;
        trans_match << 0.616895, -0.787045, 107.473759, 0.787045, 0.616895, 23.973909, 0.000000, 0.000000, 1.000000;

        for (int i = 0; i < 4; i++)
        {
            trans[i] = trans_match * trans[i];
            trans[i] = trans_change[i] * trans[i];
        }
    }

    template <typename T>
    std::vector<T> combine_profile(const std::vector<std::vector<float>> &data,
                                   std::array<Eigen::Matrix3d, 4> trans)
    {
        assert(data.size() == 4);

        std::vector<T> points;
        points.reserve(5000);

        change_trans(trans);

        for (int i = 0; i < data.size(); i++)
        {
            const auto &t = trans[i];

            double a00 = t(0, 0);
            double a01 = t(0, 1);
            double a10 = t(1, 0);
            double a11 = t(1, 1);
            double trans_x = t(0, 2);
            double trans_y = t(1, 2);

            if (data[i].size() < 2)
                continue;
            float pre_z = data[i][0];
            float z = data[i][1];
            float next_z = 0;
            for (int j = 2; j < data[i].size(); j++)
            {
                float x = j * PROFILE_X_STEP;
                next_z = data[i][j];
                if (z > PROFILE_Z_MIN_CODE && std::abs(next_z - z) < 3.0 && std::abs(z - pre_z) < 3.0)
                {
                    double x_ = x * a00 + z * a01 + trans_x;
                    double z_ = x * a10 + z * a11 + trans_y;
                    points.emplace_back(x_, z_);
                    // LOG_OUT(x_, z_, x, z, points.size(), i);
                }
                pre_z = z;
                z = next_z;
            }
        }
        return points;
    }

    std::array<Eigen::Matrix3d, 4> make_trans()
    {
        std::array<Eigen::Matrix3d, 4> trans;

        trans[0] << -0.995008, -0.099796, -359.904, 0.099796, -0.995008, -14.1822, 0, 0, 1;
        trans[1] << 0.988899, 0.148592, 267.636, -0.148592, 0.988899, 24.427, 0, 0, 1;
        trans[2] << -0.0818567, 0.996644, 13.4353, -0.996644, -0.0818567, -335.011, 0, 0, 1;
        trans[3] << -0.106439, -0.994319, -108.513, 0.994319, -0.106439, 285.097, 0, 0, 1;

        FILE* fp = NULL;
        fopen_s(&fp, "D:\\work\\test_modulse\\data\\trans_matrix_3x3.back.bin", "rb");
        double temp[36];
        fread(temp, sizeof(double), 36, fp);

        for (int k = 0; k < 4; k++)
        {
            for (int i = 0; i < 3; i++)
            {
                for (int j = 0; j < 3; j++)
                {
                    trans[k](i, j) = temp[k * 9 + i * 3 + j];
                    // std::cout<< trans[k](i, j) << ",";
                }
                // std::cout<< "\n";
            }
            // std::cout << "\n";
        }

        fclose(fp);
        return trans;
    }

    template <typename PointT>
    std::vector<PointT> get_one_profile()
    {
        auto trans = make_trans();
        const std::array<std::string, 4> file_name = {"D:\\work\\test_modulse\\data\\LaserData.std\\int3200_device=0.dat",
                                                      "D:\\work\\test_modulse\\data\\LaserData.std\\int3200_device=1.dat",
                                                      "D:\\work\\test_modulse\\data\\LaserData.std\\int3200_device=2.dat",
                                                      "D:\\work\\test_modulse\\data\\LaserData.std\\int3200_device=3.dat"};

        std::vector<std::vector<float>> data(4);
        for (int i = 0; i < 4; i++)
            data[i] = get_data_array(file_name[i], 0, 3200);

        auto points = combine_profile<PointT>(data, trans);
        return points;
    }

    template <typename PointT>
    std::vector<PointT> get_one_profile1()
    {
        auto trans = make_trans();

        std::vector<std::vector<float>> data = get_data_measure();
        auto points = combine_profile<PointT>(data, trans);
        return points;
    }
}
