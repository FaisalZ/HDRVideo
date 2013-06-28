#ifndef MATCHER_H
#define MATCHER_H
#include <QCoreApplication>
#include <QTextStream>
#include "opencv2/opencv.hpp"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <QFile>

class matcher
{
public:
    static void find_Homography(QStringList under_list, QStringList over_list, QTextStream &out, cv::Mat &H, int steps, double sift_gamma, int iso, double feature_t, double nndr);
};

#endif // MATCHER_H
