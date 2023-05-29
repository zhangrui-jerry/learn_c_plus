#include "test.hpp"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>


int test()
{
    cv::Mat image;
    image = cv::imread("./1.png");
    if (image.data == nullptr)
        std::cout << " not find image " << std::endl;
    return 0;
}