#include "controller.h"
#include <QCoreApplication>
#include <QTime>
#include <QFile>
#include <QStringList>
#include <iostream>
#include <matcher.h>
#include "opencv2/opencv.hpp"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <QTextStream>
#include <renderer.h>
#include <QDir>

double controller::start(QStringList arguments)
{
    //start timer
    double t = (double)cv::getTickCount();

    //parse arguments
    if(arguments[1]=="-find-homography")
    {
        try
        {
            QString under = arguments[2];
            QString over = arguments[3];
            QString csv_path = arguments[4];
            QString method = arguments[5];
            double feature_t = arguments[6].toDouble();
            double nndr = arguments[7].toDouble();
            double sift_gamma = arguments[8].toDouble();
            double brightness_offset = arguments[9].toDouble();
            int iso = arguments[10].toInt();
            matcher::find_Homography(under, over, csv_path, sift_gamma, iso, feature_t, nndr, method, brightness_offset);
        }
        catch(...)
        {
            std::cout << "Parameter of '-find-homography' unreadable. Type -help for a list of possible parameters."<< std::endl;
        }
    }
    else if(arguments[1]=="-merge")
    {
        try
        {
            QString under = arguments[2];
            QString over = arguments[3];
            cv::Mat homography(3,3,6);
            for (int i = 0; i < 3; i++)
            {
                for (int j = 0; j < 3; j++)
                {
                    homography.at<double>(i,j) = arguments[(i*3)+j+4].toDouble();
                }
            }
            QString out = arguments[13];
            double brightness_offset = arguments[14].toDouble();
            int iso = arguments[15].toInt();
            double lower = arguments[16].toDouble();
            double upper = arguments[17].toDouble();
            Renderer::render(under,over,upper,lower,homography,out,iso,brightness_offset,true);
        }
        catch(...)
        {
            std::cout << "Parameter of '-merge' unreadable. Type -help for a list of possible parameters."<< std::endl;
        }
    }
    else if(arguments[1]=="-transform")
    {
        try
        {
            QString image = arguments[2];
            cv::Mat homography(3,3,6);
            for (int i = 0; i < 3; i++)
            {
                for (int j = 0; j < 3; j++)
                {
                    homography.at<double>(i,j) = arguments[(i*3)+j+3].toDouble();
                }
            }
            QString out = arguments[12];
            int iso = arguments[13].toInt();
            Renderer::render(image,"0",0,0,homography,out,iso,0,false);
        }
        catch(...)
        {
            std::cout << "Parameter of '-transform' unreadable. Type -help for a list of possible parameters."<< std::endl;
        }
    }
    else if(arguments[1]=="-help" || arguments[1]=="-h")
    {
        std::cout << "HDR Merger Help:" << std::endl;
        std::cout << std::endl;
        std::cout << "Parameter:" << std::endl;
        std::cout << std::endl;
        std::cout << "-find-homography" << std::endl;
        std::cout << "  Image_Input_Lowlight_Preserving_Path" << std::endl;
        std::cout << "  Image_Input_Highlight_Preserving_Path" << std::endl;
        std::cout << "  CSV_Output_Path" << std::endl;
        std::cout << "  Method (currently only 'sift')" << std::endl;
        std::cout << "  Params(feat_tres, neares_neighbour_distance_ratio, gamma, brightness_multikator_in_lin_light , ISO)" << std::endl;
        std::cout << "  (All seperated by a space.)" << std::endl;
        std::cout << std::endl;
        std::cout << "-merge" << std::endl;
        std::cout << "  Image_Input_Lowlight_Preserving_Path" << std::endl;
        std::cout << "  Image_Input_Highlight_Preserving_Path" << std::endl;
        std::cout << "  Homography Matrix (each element seperated with space)" << std::endl;
        std::cout << "  Merged_Image_Output_path" << std::endl;
        std::cout << "  brightness_multiplikator_in_lin_light" << std::endl;
        std::cout << "  ISO (only '800' and ‘1600‘ supported)" << std::endl;
        std::cout << "  LowerBlendTreshold" << std::endl;
        std::cout << "  UpperBlendTreshold" << std::endl;
        std::cout << std::endl;
        std::cout << "-transform" << std::endl;
        std::cout << "  Image_Input_Path" << std::endl;
        std::cout << "  Homography Matrix (each element seperated with space)" << std::endl;
        std::cout << "  Transformed_Image_Output_Path" << std::endl;
        std::cout << "  ISO (only '800' and ‘1600‘ supported)" << std::endl;
    }
    else
    {
        std::cout << "Parameter unknown. Type -help for a list of possible functions."<< std::endl;
    }

    double timer = ((double)(((int)(100.0*(((double)cv::getTickCount() - t)/cv::getTickFrequency())))/100.0));
    return timer;
}
