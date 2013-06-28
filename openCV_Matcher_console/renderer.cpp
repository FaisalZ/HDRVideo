#include "renderer.h"
#include "opencv2/opencv.hpp"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <QStringList>
#include <QFileInfo>
#include <fileio.h>
#include <img_ops.h>
#include <iostream>

void Renderer::render(QString under_path, QString over_path, double upper, double lower, cv::Mat H, QString save_path, int iso)
{
        cv::Mat over = FileIO::read_linear(over_path,10,iso);
        cv::Mat under = FileIO::read_linear(under_path,1,iso);

        cv::warpPerspective(over,over,H,cv::Size(under.cols,under.rows));

        cv::Mat linear = img_ops::blend_images(under,over,(float)upper,(float)lower);
        //loga.convertTo(loga,18,1);

        QString linear_filename;

        linear_filename = save_path+"/"+QFileInfo(over_path).completeBaseName().right(27)+"_lin.exr";

        cv::imwrite(linear_filename.toLatin1().data(),linear);
}
