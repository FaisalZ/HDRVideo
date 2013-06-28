#include "linear.h"
#include "opencv2/opencv.hpp"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <qmath.h>
#include <QMessageBox>

/*Method that reads an png image and converts it into linear
 *@param    &path       path to image
 *@return   cv::Mat     linearized image
 */
cv::Mat Linear::read_linear(QString path, double shift, double &min, double &max)
{
    min = DBL_MAX;
    max = -DBL_MAX;
    cv::Mat pic = cv::imread(path.toLatin1().data(), -1);
    cv::Mat result(pic.rows,pic.cols,CV_32FC3);
    QString identifer = path.right(31);
    identifer = identifer.left(7);
    int iso = 0;
    double lowerClip, upperClip,a,b,c,d,e,f;
    if(identifer == "001C003" ||
       identifer == "001C011" ||
       identifer == "005C003" ||
       identifer == "005C005")
    {
        iso = 1600;
        lowerClip = 0.013047;
        upperClip = 1023/1023;
        a = 5.555556;
        b = 0.038625;
        c = 0.237781;
        d = 0.387093;
        e = 5.163350;
        f = 0.092824;
    }
    if(identifer == "002C001" ||
       identifer == "002C004" ||
       identifer == "002C006" ||
       identifer == "002C012" ||
       identifer == "003C005" ||
       identifer == "004C003" ||
       identifer == "004C011" ||
       identifer == "004C020" ||
       identifer == "004C021" ||
       identifer == "005C003" ||
       identifer == "005C005" ||
       identifer == "003C005" ||
       identifer == "006C003" ||
       identifer == "006C008")
    {
        iso = 800;
        lowerClip = 0.010591;
        upperClip = 1023/1023;
        a = 5.555556;
        b = 0.052272;
        c = 0.247190;
        d = 0.385537;
        e = 5.367655;
        f = 0.092809;
    }

    if(iso == 0)
    {
        QMessageBox msgBox;
        msgBox.setText("Unable to identify iso value. Taking 800.");
        msgBox.exec();
        iso = 800;
        lowerClip = 0.010591;
        upperClip = 1023/1023;
        a = 5.555556;
        b = 0.052272;
        c = 0.247190;
        d = 0.385537;
        e = 5.367655;
        f = 0.092809;
    }

    {
        //calculate LUT
        QVector<float> color_lut(65536);
        float lowest = FLT_MAX;
        float highest = -FLT_MAX;
        float index = e*lowerClip+f;
        for(int i = 0; i < 65536;i++)
        {
            float val = ((float)i)/(float)65535;
            float out;
            if(val > upperClip)
            {
                val = upperClip;
            }
            if(val > index)
            {
                out = (qPow((float)10,((val-d)/c))-b)/a;
            }
            else
            {
                out = (val-f)/e;
            }
            out = out/shift;
            color_lut[i] = out;
        }

        //Apply LUT
        for(int y = 0; y < pic.rows; y++) //height
        {
            for(int x = 0; x < pic.cols; x++) //width
            {
                for(int rgb = 0; rgb < pic.channels(); rgb++) //colorchannel
                {
                    ushort val_u = ((ushort*)(pic.data))[ y*pic.step1()+ x*pic.channels() + rgb];
                    ((float*)(result.data))[y*result.step1()+ x*result.channels() + rgb] = color_lut[val_u];
                    if(color_lut[val_u] < min) min = color_lut[val_u];
                    if(color_lut[val_u] > max) max = color_lut[val_u];
                }
            }
        }
    }
    return result;
}
