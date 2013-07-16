#include "renderer.h"
#include "opencv2/opencv.hpp"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <QStringList>
#include <QFileInfo>
#include <fileio.h>
#include <img_ops.h>
#include <iostream>

void Renderer::render(QString under_path, QString over_path, double upper, double lower, cv::Mat H, QString save_path, int iso, double brightness_offset,bool merge)
{
    if(merge)
    {
        cv::Mat over = FileIO::read_linear(over_path,10,iso);
        cv::Mat under = FileIO::read_linear(under_path,1,iso);

        cv::warpPerspective(under,under,H,cv::Size(under.cols,under.rows));

        cv::Mat linear = img_ops::blend_images(under,over,(float)upper,(float)lower);

        QString linear_filename;

        linear_filename = save_path+"/"+under_path.right(11).left(7)+"_lin.exr";

        cv::imwrite(linear_filename.toLatin1().data(),linear);
    }
    else
    {
        cv::Mat under = cv::imread(under_path.toLatin1().data(), -1);
        cv::warpPerspective(under,under,H,cv::Size(under.cols,under.rows));

        QString filename;
        filename = save_path+"/"+QFileInfo(under_path).completeBaseName().right(28)+"_transformed.png";

        cv::imwrite(filename.toLatin1().data(),under);
    }
}
