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
    static void find_Homography(QString under, QString over, QString out_path, double sift_gamma, int iso, double feature_t, double nndr, QString method, double brightness_offset);
};

#endif // MATCHER_H
