#ifndef RENDERER_H
#define RENDERER_H

#include "opencv2/opencv.hpp"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <QStringList>

class Renderer
{
public:
    static void render(QString under, QString over, double upper, double lower, cv::Mat H, QString save_path, int iso);
};

#endif // RENDERER_H
