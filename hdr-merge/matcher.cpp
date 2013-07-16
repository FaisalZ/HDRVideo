#include "matcher.h"
#include <QCoreApplication>
#include <QFile>
#include <QTextStream>
#include <QStringList>
#include "opencv2/opencv.hpp"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/flann/kdtree_index.h>
#include <opencv2/stitching/detail/matchers.hpp>
#include <QTime>
#include <qmath.h>
#include <fileio.h>
#include <iostream>
#include <QVector>
#include <QDir>

void matcher::find_Homography(QString under, QString over, QString out_path, double sift_gamma, int iso, double feature_t, double nndr, QString method, double brightness_offset)
{
    //initialise needed Variables
    //keypoints
    std::vector< cv::KeyPoint > keypoints_under;
    std::vector< cv::KeyPoint > keypoints_over;
    //Keypoint Descriptors
    cv::Mat desc_under,desc_over;
    //vectors holding the two best nearest neighbors per descriptor
    std::vector< std::vector< cv::DMatch > > matches1;
    std::vector< std::vector< cv::DMatch > > matches2;
    //vector holding the best nearest neighbor per descriptor
    std::vector< cv::DMatch > good_matches;
    //vector holding ponts of matches
    std::vector< cv::Point2f > matchpoints_1,matchpoints_2;
    //Homography matrix
    cv::Mat H;

    QDir dir(out_path);
    bool s = dir.mkpath(out_path);
    QFile file(out_path+"/Homographys.csv");
    file.open(QIODevice::Append);
    QTextStream out(&file);
    out << under.right(11).left(7)+": ";

    //load two images
    cv::Mat under_img = FileIO::read_linear(under,1,iso);
    cv::pow(under_img,(1.0f/sift_gamma),under_img);
    under_img.convertTo(under_img, CV_8U, 256);
    cv::Mat over_img  = FileIO::read_linear(over,brightness_offset,iso);
    cv::pow(over_img,(1.0f/sift_gamma),over_img);
    over_img.convertTo(over_img, CV_8U, 256);

    //detect features
    //if(method=="sift")
    if(method=="sift")
    {
        std::cout << "sift" << std::endl;
        cv::SiftFeatureDetector sifter(feature_t);
        sifter.detect(under_img,keypoints_under);
        sifter.detect(over_img,keypoints_over);

        //run Descriptor Extractor in keypoints
        cv::SiftDescriptorExtractor siftDesc;
        siftDesc.compute(under_img,keypoints_under,desc_under);
        siftDesc.compute(over_img,keypoints_over,desc_over);
    }
    else if(method=="sift2")
    {
        std::cout << "sift2" << std::endl;
        cv::Ptr<cv::FeatureDetector> detector = cv::FeatureDetector::create("SIFT");
        detector->detect(under_img,keypoints_under);
        detector->detect(over_img,keypoints_over);

        cv::Ptr<cv::DescriptorExtractor> desc = cv::DescriptorExtractor::create("SIFT");
        desc->compute(under_img,keypoints_under,desc_under);
        desc->compute(over_img,keypoints_over,desc_over);
    }
    else if(method=="freak")
    {
        std::cout << "freak" << std::endl;
        cv::Ptr<cv::FeatureDetector> detector = cv::FeatureDetector::create("SIFT");
        detector->detect(under_img,keypoints_under);
        detector->detect(over_img,keypoints_over);

        cv::Ptr<cv::DescriptorExtractor> desc = cv::DescriptorExtractor::create("FREAK");

        desc->compute(under_img,keypoints_under,desc_under);
        desc->compute(over_img,keypoints_over,desc_over);
    }

    std::cout << "Keypoints found in darker img: " << QString::number(keypoints_under.size()).toLatin1().data() << std::endl;
    std::cout << "Keypoints found in brighter img: " << QString::number(keypoints_over.size()).toLatin1().data() << std::endl;

    if(keypoints_under.size() > 20 && keypoints_over.size() > 20)
    {
        //run BruteForce feature matcher
        //the two 2 nearest neighbors per descriptor will be found
        cv::BFMatcher matcher(cv::NORM_L2,false);
        //matches img1 -> img2
        matcher.knnMatch(desc_over,desc_under,matches1,2);
        //matches img2 -> img1
        matcher.knnMatch(desc_under,desc_over,matches2,2);

        std::cout << "Matches found a->b: " << QString::number(matches1.size()).toLatin1().data() << std::endl;
        std::cout << "Matches found b->a: " << QString::number(matches2.size()).toLatin1().data() << std::endl;

        //determine the best nearest neighbor match for matches1
        for (size_t i = 0; i < matches1.size(); ++i)
        {
            if (matches1[i].size() > 1)
            {
                if(matches1[i][0].distance/matches1[i][1].distance > nndr)
                {
                    matches1[i].clear();
                }
            }
            else
            {
                matches1[i].clear();
            }
        }
        //determine the best nearest neighbor match for matches2
        for (size_t i = 0; i < matches2.size(); ++i)
        {
            if (matches2[i].size() > 1)
            {
                if(matches2[i][0].distance/matches2[i][1].distance > nndr)
                {
                    matches2[i].clear();
                }
            }
            else
            {
                matches2[i].clear();
            }
        }

        //perform symetric check
        for (size_t i = 0; i < matches1.size(); ++i)
        {
            if(matches1[i].size() < 2)
                continue;
            for (size_t j = 0; j < matches2.size(); ++j)
            {
                if(matches2[j].size() < 2)
                    continue;
                if(matches1[i][0].queryIdx == matches2[j][0].trainIdx && matches2[j][0].queryIdx == matches1[i][0].trainIdx)
                {
                    good_matches.push_back(cv::DMatch(matches1[i][0].queryIdx,matches1[i][0].trainIdx,matches1[i][0].distance));
                    break;
                }
            }
        }

        std::cout << "Good matches: " << QString::number(good_matches.size()).toLatin1().data() <<std::endl;

        //get corresponding points
        for( int i = 0; i < good_matches.size(); i++ )
        {
            //-- Get the keypoints from the good matches
            matchpoints_1.push_back( keypoints_over[ good_matches[i].queryIdx ].pt );
            matchpoints_2.push_back( keypoints_under[ good_matches[i].trainIdx ].pt );
        }

        if(good_matches.size() > 20)
        {
            H = cv::findHomography(matchpoints_2,matchpoints_1,CV_RANSAC);
            //write to csv file
            for (int i = 0; i < 3; i++)
            {
                for (int j = 0; j < 3; j++)
                {
                    out << QString::number(H.at<double>(i,j))+" ";
                }
            }
            out << "\n";
        }
        else
        {
            out << "Ignored frame (Not enough good matches)\n";
        }
    }
    else
    {
        out << "Ignored frame (Not enough keypoints.)\n";
    }
    file.close();
}
