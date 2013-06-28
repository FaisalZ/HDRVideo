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

double controller::start(QString path)
{
    //start timer
    double t = (double)cv::getTickCount();
    //data holding
    QStringList under_list, over_list;
    QString save_path;
    double feature_t, nndr, lower, upper, sift_gamma;
    int iso, steps;

    std::cout << "Reading arguments from file..." << std::endl;
    FileIO::read_csv(path, under_list, over_list, iso, save_path, feature_t, nndr, lower, upper, sift_gamma, steps);
    int max;
    (over_list.size() < under_list.size()) ? max = over_list.size() : max = under_list.size();

    //bool s = QDir::mkpath(save_path);
    QDir dir(save_path);
    bool s = dir.mkpath(save_path);

    std::cout << "Save path: " << save_path.toLatin1().data() <<std::endl;
    std::cout << "Iso value: " << iso <<std::endl;
    std::cout << "Feature treshhold: " << feature_t <<std::endl;
    std::cout << "Nearest neighbor distance ratio: " << nndr <<std::endl;
    std::cout << "Lower blending treshhold: " << lower <<std::endl;
    std::cout << "Upper blending treshhold: " << upper <<std::endl;
    std::cout << "Averaging homography steps: " << steps <<std::endl;
    std::cout << "Gamma for SIFT: " << sift_gamma <<std::endl;
    std::cout << "# Frames to process: " << max <<std::endl;
    std::cout << std::endl;

    //create a Log txt file
    QDate date = QDate::currentDate();
    QString dateString = date.toString("dd-MM-yyyy");
    QTime time = QTime::currentTime();
    QString timeString = time.toString("hh:mm:ss");
    QFile file(save_path+"/Log_"+dateString+"_"+timeString+".txt");
    file.open(QIODevice::WriteOnly);
    QTextStream out (&file);
    out << "Logfile of HDR rendering \n";
    out << "Date: "+dateString+" \n";
    out << "Date: "+timeString+" \n\n";
    out << "Save path: "+save_path+" \n";
    out << "Iso value: "+QString::number(iso)+" \n";
    out << "Feature treshhold: "+QString::number(feature_t)+" \n";
    out << "Nearest neighbor distance ratio: "+QString::number(nndr)+" \n";
    out << "Lower blending treshhold: "+QString::number(lower)+" \n";
    out << "Upper blending treshhold: "+QString::number(upper)+" \n";
    out << "Averaging homography steps: "+QString::number(steps)+" \n";
    out << "Gamma for SIFT: "+QString::number(sift_gamma)+" \n";
    out << "# Frames to process: "+QString::number(max)+" \n\n";

    std::cout << "Calculating homography..." <<std::endl;
    cv::Mat H;
    matcher::find_Homography(under_list, over_list, out, H, steps, sift_gamma, iso, feature_t, nndr);
    std::cout << std::endl;
    std::cout << "Homography found." <<std::endl;
    std::cout <<std::endl;
    out << "\n----------------------------------\n";
    out << "-----final homography matrix------\n";
    out << QString::number(H.at<double>(0,0))+" "+QString::number(H.at<double>(0,1))+" "+QString::number(H.at<double>(0,2))+"\n";
    out << QString::number(H.at<double>(1,0))+" "+QString::number(H.at<double>(1,1))+" "+QString::number(H.at<double>(1,2))+"\n";
    out << QString::number(H.at<double>(2,0))+" "+QString::number(H.at<double>(2,1))+" "+QString::number(H.at<double>(2,2))+"\n";
    out << "---------------------------------- \n";

    std::cout << "Rendering Images..." <<std::endl;
    for(int i = 0; i < max; i++)
    {
        Renderer::render(under_list[i], over_list[i], upper, lower, H, save_path, iso);
                std::cout << "|" << std::flush;
    }
    std::cout << std::endl;

    double timer = ((double)(((int)(100.0*(((double)cv::getTickCount() - t)/cv::getTickFrequency())))/100.0));
    std::cout << "Seconds needed for scene: " << timer << std::endl;
    std::cout << "------------ Scene finished. ------------"<< std::endl;
    file.close();
    return timer;
}
