#include "fileio.h"
#include <QCoreApplication>
#include <QFile>
#include <QStringList>
#include <QTextStream>
#include <iostream>
#include "opencv2/opencv.hpp"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <qmath.h>
#include <QString>
#include <QVector>

void FileIO::read_csv(QString path, QStringList &under_list, QStringList &over_list, int &iso, QString &save_path, double &feature_t, double &nndr, double &lower, double &upper, double &sift_gamma, int &steps)
{
    QFile file(path);
    QString line;
    if (file.open(QIODevice::ReadOnly))
    {
        QTextStream stream( &file );
        bool first = true;
        bool second = true;
        int counter = 0;
        while (!stream.atEnd())
        {
            line = stream.readLine();
            QStringList line_list = line.split(" ");
            if(first)
            {
                iso = line_list[0].toInt();
                feature_t = line_list[1].toDouble();
                nndr = line_list[2].toDouble();
                lower = line_list[3].toDouble();
                upper = line_list[4].toDouble();
                steps = line_list[5].toInt();
                sift_gamma = line_list[6].toDouble();
                first = false;
            }
            else
            {
                if(second)
                {
                    save_path = line_list[0];
                    second = false;
                }
                else
                {
                    under_list.append(line_list[0]);
                    over_list.append(line_list[1]);
                }
            }
            counter++;
        }
    }
    file.close();
}

/*Method that reads an png image and converts it into linear
 *@param    &path       path to image
 *@return   cv::Mat     linearized image
 */
cv::Mat FileIO::read_linear(QString path, double shift, int iso)
{
    cv::Mat pic = cv::imread(path.toLatin1().data(), -1);
    cv::Mat result(pic.rows,pic.cols,CV_32FC3);
    QString identifer = path.right(31);
    identifer = identifer.left(7);
    double lowerClip, upperClip,a,b,c,d,e,f;
    if(iso == 1600)
    {
        lowerClip = 0.013047;
        upperClip = 1023/1023;
        a = 5.555556;
        b = 0.038625;
        c = 0.237781;
        d = 0.387093;
        e = 5.163350;
        f = 0.092824;
    }
    if(iso == 800)
    {
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
        std::cout << "Unable to identify ISO" << std::endl;
    }
    else
    {
        //calculate LUT
        QVector<float> color_lut(65536);
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
                }
            }
        }
    }
    return result;
}

