#pragma once
#include <fstream>
#include <iostream>
#include <vector>
#include <string>
#include "opencv2/core.hpp"
#include <stdio.h>


#define PROFILE_X_STEP 0.1
#define PROFILE_Z_MIN_CODE -900


namespace dataset
{
    std::vector<float> get_data_array(const std::string& file_name, int offset, int size)
    {
        std::ifstream file(file_name, std::ios::binary);
        if (!file) 
            std::cout << "!file" << std::endl;

        file.seekg(offset, std::ios::beg);

        std::vector<int> buffer(size);
        file.read((char *)buffer.data(), sizeof(int) * size);

        std::vector<float> data(size);
        for (int i = 0; i < size; i++)
            data[i] = (float)(buffer[i] / 100000.0);

        return data;
    }

    template <typename T>
    std::vector<T> convert_points(const std::vector<float>& data)
    {
        std::vector<T> points;
        for (int i = 0; i < data.size(); i++)
        {
            float z = data[i];
            float x = i * PROFILE_X_STEP;
            if (z > PROFILE_Z_MIN_CODE)
                points.emplace_back(x, z);
        }
        return points;
    }
}