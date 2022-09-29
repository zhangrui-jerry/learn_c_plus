#pragma once 
#include <vector>
#include <fstream>
#include <string>
#include <iostream>

#define PROFILE_X_STEP -0.2
#define PROFILE_Z_MIN_CODE -900

namespace dataset
{
    std::vector<float> get_data_array(const std::string& file_name, int offset, int size)
    {
        std::vector<float> data(size);
        FILE *fp;
        fopen_s(&fp, file_name.c_str(), "rb");
        fseek(fp, offset, SEEK_SET);
        float buffer[3200];
        int* buf = (int*)buffer;
        fread(buf, sizeof(float), size, fp);
        for (int i = 0; i < size; i++)
        {
            float temp = (float)(buf[i] / 100000.0);
            data[i] = temp;
        }
        fclose(fp);
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

    template <typename T>
    std::vector<T> generate_circle(float radius = 1.0, float step = 0.01)
    {
        std::vector<T> points;
        float angle_step = step / radius;
        for (float angle = 0.0; angle < 2 * 3.1415926; angle += angle_step)
        {
            float x = radius * std::cos(angle);
            float y = radius * std::sin(angle);
            points.emplace_back(x, y);
        }
        return points;
    }
}