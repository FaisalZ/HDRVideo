#ifndef FILEIO_H
#define FILEIO_H
#include <QCoreApplication>
#include "opencv2/opencv.hpp"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>

class FileIO
{
public:
    static void read_csv(QString path, QStringList &under_list, QStringList &over_list, int &iso, QString &save_path, double &feature_t, double &nndr, double &lower, double &upper, double &sift_gamma, int &steps);
    static cv::Mat read_linear(QString path, double shift, int iso);
};

#endif // FILEIO_H
