#pragma once
#include "opencv2/opencv.hpp"
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include <vector>
#include <string>
#include "ellipese/dataset.hpp"
#include "mylog/mylog.hpp"
#include "ransac/ransac.hpp"

namespace fit
{
    std::vector<cv::Point2f> make_data()
    {
#if 0
        std::vector<cv::Point2f>pts;
        using namespace cv;
        pts.push_back(Point2f(173.41854895999165f, 125.84473135880411f));
        pts.push_back(Point2f(180.63769498640912f, 130.960006577589f));
        pts.push_back(Point2f(174.99173759130173f, 137.34265632926764f));
        pts.push_back(Point2f(170.9044645313217f, 141.68017556480243f));
        pts.push_back(Point2f(163.48965388499656f, 141.9404438924043f));
        pts.push_back(Point2f(159.37687818401147f, 148.60835331594876f));
        pts.push_back(Point2f(150.38917629356735f, 155.68825577720446f));
        pts.push_back(Point2f(147.16319653316862f, 157.06039984963923f));
        pts.push_back(Point2f(141.73118707843207f, 157.2570155198414f));
        pts.push_back(Point2f(130.61569602948597f, 159.40742182929364f));
        pts.push_back(Point2f(127.00573042229027f, 161.34430232187867f));
        pts.push_back(Point2f(120.49383815053747f, 163.72610883128334f));
        pts.push_back(Point2f(114.62383760040998f, 162.6788666385239f));
        pts.push_back(Point2f(108.84871269183333f, 161.90597054388132f));
        pts.push_back(Point2f(103.04574087829076f, 167.44352944383985f));
        pts.push_back(Point2f(96.31623870161255f, 163.71641295746116f));
        pts.push_back(Point2f(89.86174417295126f, 157.2967811253635f));
        pts.push_back(Point2f(84.27940674801192f, 168.6331304010667f));
        pts.push_back(Point2f(76.61995117937661f, 159.4445412678832f));
        pts.push_back(Point2f(72.22526316142418f, 154.60770776728293f));
        pts.push_back(Point2f(64.97742405067658f, 152.3687174339018f));
        pts.push_back(Point2f(58.34612797237003f, 155.61116802371583f));
        pts.push_back(Point2f(55.59089117268539f, 148.56245696566418f));
        pts.push_back(Point2f(45.22711195983706f, 145.6713241271927f));
        pts.push_back(Point2f(40.090542298840234f, 142.36141304004002f));
        pts.push_back(Point2f(31.788996807277414f, 136.26164877915585f));
        pts.push_back(Point2f(27.27613006088805f, 137.46860042141503f));
        pts.push_back(Point2f(23.972392188502226f, 129.17993872328594f));
        pts.push_back(Point2f(20.688046711616977f, 121.52750840733087f));
        pts.push_back(Point2f(14.635115184257643f, 115.36942800110485f));
        pts.push_back(Point2f(14.850919318756809f, 109.43609786936987f));
        pts.push_back(Point2f(7.476847697758103f, 102.67657265589285f));
        pts.push_back(Point2f(1.8896944088091914f, 95.78878215565676f));
        pts.push_back(Point2f(1.731997022935417f, 88.17674033990495f));
        pts.push_back(Point2f(1.6780841363402033f, 80.65581939883002f));
        pts.push_back(Point2f(0.035330281415411946f, 73.1088693846768f));
        pts.push_back(Point2f(0.14652518786238033f, 65.42769523404296f));
        pts.push_back(Point2f(6.99914645302843f, 58.436451064804245f));
        pts.push_back(Point2f(6.719616410428614f, 50.15263031354927f));
        pts.push_back(Point2f(5.122267598477748f, 46.03603214691343f));

#else
        auto data = dataset::get_data_array("D:\\work\\modulse\\data\\LaserData\\int800x2_device=1.dat", 800 * sizeof(float), 800);
        auto pts = dataset::convert_points<cv::Point2f>(data);
#endif

        return pts;
    }

        void draw_result(std::vector<cv::Point2f> &pts, const cv::RotatedRect &res)
    {
        if (pts.empty())
            return;

        float max_x = pts[0].x;
        float max_y = pts[0].y;
        float min_x = pts[0].x;
        float min_y = pts[0].y;

        for (const auto &point : pts)
        {
            max_x = std::max(max_x, point.x);
            max_y = std::max(max_y, point.y);
            min_x = std::min(min_x, point.x);
            min_y = std::min(min_y, point.y);
        }

        // std::cout << max_x << " "
        //           << " " << max_y << " " << min_x << " " << min_y << std::endl;

        const int ratio = 20;
        const int win_width = 10;

        auto fx = [min_x, ratio, win_width](float val)
        { return (int)((val - min_x + win_width) * ratio); };
        auto fy = [min_y, ratio, win_width](float val)
        { return (int)((val - min_y + win_width) * ratio); };

        cv::Vec3b color{(uchar)(rand() % 255), (uchar)(rand() % 255), (uchar)(rand() % 255)};
        cv::Mat img(cv::Size(fx(max_x) + win_width * ratio, fy(max_y) + win_width * ratio), CV_8UC3, cv::Scalar::all(255));

        for (const auto &point : pts)
            cv::circle(img, {fx(point.x), fy(point.y)}, 2, color, 3);

        auto r = res;
        r.size.height = r.size.height * ratio;
        r.size.width = r.size.width * ratio;
        r.center.x = (r.center.x - min_x + win_width) * ratio;
        r.center.y = (r.center.y - min_y + win_width) * ratio;

        cv::circle(img, r.center, 1, color, 2);
        cv::ellipse(img, r, cv::Scalar(0, 0, 255), 1, cv::LINE_AA);

        cv::namedWindow("img", 0);
        cv::imshow("img", img);
        cv::waitKey(0);
    }

    void test(int argc, char** argv)
    {
        std::string file_name = "D:\\work\\modulse\\data\\LaserData\\int800x2_device=0.dat";;
        int offset = 0;
        if (argc == 2)
        {
            file_name = argv[1];
            LOG_OUT(file_name);
        }
        if (argc == 3)
        {
            file_name = argv[1];
            offset = std::stoi(argv[2]);
            LOG_OUT(file_name);
        }

        int n = 20;
        for (int i = 0; i < n; i++)
        {
            auto data = dataset::get_data_array(file_name, 1600 * sizeof(float) * i + offset, 800);
            auto pts = dataset::convert_points<cv::Point2f>(data);

            cv::RotatedRect ellipseAMSTest = cv::fitEllipseDirect(pts);
            auto circle = ransac::fit_circle(pts);
            auto ellipese = ransac::fit_ellipese(pts);

            LOG_ADD("ellipseAMSTest.size.width",  ellipseAMSTest.size.width);
            LOG_ADD("ellipseAMSTest.size.height", ellipseAMSTest.size.height);
            LOG_ADD("circle.radius", circle.radius);
            LOG_ADD("circle.center[0]", circle.center[0]);
            LOG_ADD("circle.center[1]", circle.center[1]);
            LOG_ADD("ellipese.width", ellipese.width);
            LOG_ADD("ellipese.height", ellipese.height);
            LOG_ADD("ellipese.center[0]", ellipese.center[0]);
            LOG_ADD("ellipese.center[1]", ellipese.center[1]);
            // LOG_OUT("========= circle", circle.radius, circle.center);
            // draw_result(pts, ellipseAMSTest);
            // LOG_OUT(ellipseAMSTest.size, circle.radius);
        }

        // std::cout << "==================" << std::endl;

        // for (int i = 0; i < n; i++)
        // {
        //     auto data = dataset::get_data_array("D:\\work\\modulse\\data\\LaserData\\int800x2_device=0.dat", 1600 * sizeof(float) * i + 800, 800);
        //     auto pts = dataset::convert_points<cv::Point2f>(data);
        //     cv::RotatedRect ellipseAMSTest = cv::fitEllipse(pts);
        //     draw_result(pts, ellipseAMSTest);
        //     std::cout << ellipseAMSTest.size << " " << ellipseAMSTest.angle << " " << ellipseAMSTest.center << std::endl;
        // }

        // std::cout << "==================" << std::endl;

        // for (int i = 0; i < n; i++)
        // {
        //     auto data = dataset::get_data_array("D:\\work\\modulse\\data\\LaserData\\int800x2_device=1.dat", 1600 * sizeof(float) * i, 800);
        //     auto pts = dataset::convert_points<cv::Point2f>(data);
        //     cv::RotatedRect ellipseAMSTest = cv::fitEllipse(pts);
        //     draw_result(pts, ellipseAMSTest);
        //     std::cout << ellipseAMSTest.size << " " << ellipseAMSTest.angle << " " << ellipseAMSTest.center << std::endl;
        // }

        // std::cout << "==================" << std::endl;

        // for (int i = 0; i < n; i++)
        // {
        //     auto data = dataset::get_data_array("D:\\work\\modulse\\data\\LaserData\\int800x2_device=1.dat", 1600 * sizeof(float) * i + 800, 800);
        //     auto pts = dataset::convert_points<cv::Point2f>(data);
        //     cv::RotatedRect ellipseAMSTest = cv::fitEllipse(pts);
        //     draw_result(pts, ellipseAMSTest);
        //     std::cout << ellipseAMSTest.size << " " << ellipseAMSTest.angle << " " << ellipseAMSTest.center << std::endl;
        // }
    }

    void fit_ellipese(int argc, char** argv)
    {
        // auto pts = make_data();
        // cv::RotatedRect ellipseAMSTest = cv::fitEllipse(pts);
        // std::cout << ellipseAMSTest.size << " " << ellipseAMSTest.angle << " " << ellipseAMSTest.center << std::endl;
        // draw_result(pts, ellipseAMSTest);

        test(argc, argv);
        LOG_ALL_MEAN();
    }
}