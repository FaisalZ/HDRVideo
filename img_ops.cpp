#include "img_ops.h"
#include "opencv2/opencv.hpp"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <qmath.h>
#include <QMessageBox>

cv::Mat img_ops::blend_images(cv::Mat &img_a, cv::Mat &img_b, float upper, float lower)
{
    double low = (double)lower;
    double up = (double)upper;


    cv::Mat img = cv::Mat::zeros(img_a.rows,img_a.cols,img_a.type());

    for(int y = 0; y < img_a.rows; y++)
    {
        for(int x = 0; x < img_a.cols; x++)
        {
            for(int color = 0; color < 3; color++)
            {
                float val_o = ((float*)(img_b.data))[ y*img_b.step1()+ x*img_b.channels() + color];
                float val_u = ((float*)(img_a.data))[ y*img_a.step1()+ x*img_a.channels() + color];
                if(val_u < low)
                {
                    ((float*)(img.data))[y*img.step1()+ x*img.channels() + color] = val_o;
                }
                else if(val_u > up)
                {
                    ((float*)(img.data))[y*img.step1()+ x*img.channels() + color] = val_u;
                }
                else
                {
                    float alpha = (val_u-low)/(up-low);
                    float val = (alpha*val_o)+((1.0f-alpha)*val_u);
                    ((float*)(img.data))[y*img.step1()+ x*img.channels() + color] = val;
                }
            }
        }
    }
    return img;
}
