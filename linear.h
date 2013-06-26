#ifndef LINEAR_H
#define LINEAR_H
#include <QMainWindow>
#include "opencv2/opencv.hpp"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>

class Linear
{
public:
    static cv::Mat read_linear(QString path, double shift, double &min, double &max);
};

#endif // LINEAR_H
