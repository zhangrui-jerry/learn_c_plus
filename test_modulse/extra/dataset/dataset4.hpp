#pragma once
#include <fstream>
#include <vector>
#include <string>
#include <memory>
#include <iostream>
#include <array>

namespace dataset
{
    using ProfileArray = std::array<std::vector<float>, 4>; 
    using Real = float;

    inline ProfileArray get_data_array4(const std::string& file_name, int offset, int size)
    {
        std::ifstream file(file_name, std::ios::binary);
        if (!file) 
            std::cout << "!file" << std::endl;

        file.seekg(offset, std::ios::beg);

        ProfileArray data;
        for (int i = 0; i < 2; i++)
        {
            data[i].resize(size);
            file.read((char *)data[i].data(), sizeof(Real) * size);
        }
        file.seekg(offset + sizeof(Real) * 4 * 2 * size + sizeof(Real) * 2 * size, std::ios::beg);

        for (int i = 2; i < 4; i++)
        {
            data[i].resize(size);
            file.read((char *)data[i].data(), sizeof(Real) * size);
        }

        return data;
    }

    inline ProfileArray get_data_measure()
    {
        static int count = 6145;
        int offset = count * sizeof(int) * 3200 * 4;
        count++;
        count = count % 20000;
        return get_data_array4("../data/LaserData.m/LaserData/float_3200x4_2022-07-15_101731.000.dat", offset, 3200);
    }
};