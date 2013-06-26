#ifndef IMG_OPS_H
#define IMG_OPS_H

#include <QMainWindow>
#include "opencv2/opencv.hpp"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>

class img_ops
{
public:
    static cv::Mat blend_images(cv::Mat &img_a, cv::Mat &img_b, float upper, float lower);
};

#endif // IMG_OPS_H
