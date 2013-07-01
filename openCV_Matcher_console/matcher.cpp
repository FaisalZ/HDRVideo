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

void matcher::find_Homography(QStringList under_list, QStringList over_list, QTextStream &out, cv::Mat &H, int steps, double sift_gamma, int iso, double feature_t, double nndr)
{
    if(over_list.length() > 0 && under_list.length() > 0)
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

        //loop variables
        QVector< cv::Mat > Hom;
        QVector< int > matches_counter;

        //perform for every nth frame
        int max;
        (over_list.size() < under_list.size()) ? max = over_list.size() : max = under_list.size();
        int counter = 0;
        for(int looper = 0; looper < max; looper= looper+steps)
        {
            good_matches.clear();
            //load two images
            cv::Mat under_img = FileIO::read_linear(under_list[looper],1,iso);
            cv::pow(under_img,(1.0f/sift_gamma),under_img);
            under_img.convertTo(under_img, CV_8U, 256);
            cv::Mat over_img  = FileIO::read_linear(over_list[looper],10,iso);
            cv::pow(over_img,(1.0f/sift_gamma),over_img);
            over_img.convertTo(over_img, CV_8U, 256);

            //detect features
            cv::SiftFeatureDetector sifter(feature_t);
            sifter.detect(under_img,keypoints_under);
            sifter.detect(over_img,keypoints_over);

            //run Descriptor Extractor in keypoints
            cv::SiftDescriptorExtractor siftDesc;
            siftDesc.compute(under_img,keypoints_under,desc_under);
            siftDesc.compute(over_img,keypoints_over,desc_over);

            out << "Imagepair "+QString::number(looper)+" :\n";
            out << "Keypoints found in darker img: "+QString::number(keypoints_under.size())+"\n";
            out << "Keypoints found in brighter img: "+QString::number(keypoints_over.size())+"\n";

            if(keypoints_under.size() > 20 && keypoints_over.size() > 20)
            {
                //run BruteForce feature matcher
                //the two 2 nearest neighbors per descriptor will be found
                cv::BFMatcher matcher(cv::NORM_L2,false);
                //matches img1 -> img2
                matcher.knnMatch(desc_over,desc_under,matches1,2);
                //matches img2 -> img1
                matcher.knnMatch(desc_under,desc_over,matches2,2);

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

                out << "Good matches: "+QString::number(good_matches.size())+"\n";

                //get corresponding points
                for( int i = 0; i < good_matches.size(); i++ )
                {
                    //-- Get the keypoints from the good matches
                    matchpoints_1.push_back( keypoints_over[ good_matches[i].queryIdx ].pt );
                    matchpoints_2.push_back( keypoints_under[ good_matches[i].trainIdx ].pt );
                }

                if(good_matches.size() > 20)
                {
                    Hom.append(cv::findHomography(matchpoints_1,matchpoints_2,CV_RANSAC));
                    matches_counter.append(good_matches.size());
                    //write to Log file
                    cv::Mat p00(3,1,Hom[0].type());
                    p00.at<double>(0,0) = 0;
                    p00.at<double>(1,0) = 0;
                    p00.at<double>(2,0) = 1;
                    cv::Mat p01(3,1,Hom[0].type());
                    p01.at<double>(0,0) = 1920;
                    p01.at<double>(1,0) = 0;
                    p01.at<double>(2,0) = 1;
                    cv::Mat p10(3,1,Hom[0].type());
                    p10.at<double>(3,1) = 0;
                    p10.at<double>(1,0) = 1080;
                    p10.at<double>(2,0) = 1;
                    cv::Mat p11(3,1,Hom[0].type());
                    p11.at<double>(0,0) = 1920;
                    p11.at<double>(1,0) = 1080;
                    p11.at<double>(2,0) = 1;
                    cv::Mat p00x;
                    gemm(Hom[counter],p00,1,NULL,0,p00x);
                    cv::Mat p01x;
                    gemm(Hom[counter],p01,1,NULL,0,p01x);
                    cv::Mat p10x;
                    gemm(Hom[counter],p10,1,NULL,0,p10x);
                    cv::Mat p11x;
                    gemm(Hom[counter],p11,1,NULL,0,p11x);
                    //Output Points
                    out << "Offset of edge Pixels: \n";
                    out << QString::number(p00x.at<double>(0,0)/p00x.at<double>(2,0))+","+QString::number(p00x.at<double>(1,0)/p00x.at<double>(2,0))+",";
                    out << QString::number(p01x.at<double>(0,0)/p01x.at<double>(2,0))+","+QString::number(p01x.at<double>(1,0)/p01x.at<double>(2,0))+",";
                    out << QString::number(p10x.at<double>(0,0)/p10x.at<double>(2,0))+","+QString::number(p10x.at<double>(1,0)/p10x.at<double>(2,0))+",";
                    out << QString::number(p11x.at<double>(0,0)/p11x.at<double>(2,0))+","+QString::number(p11x.at<double>(1,0)/p11x.at<double>(2,0))+"\n\n";
                    counter++;
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
                std::cout << "|" << std::flush;
        }//end for (every 10th frame)

        //average Homography
        double sum = 0;
        for(int i = 0; i < matches_counter.size(); i++)
        {
            sum = sum+matches_counter[i];
        }
        if(Hom.size() > 0)
        {
            H = ((double)matches_counter[0]/(double)sum)*Hom[0];
            for(int i = 1; i < Hom.size() ; i++)
            {
                H = H+((double)matches_counter[i]/(double)sum)*Hom[i];
            }
        }
    }
    else
    {
        std::cout << "You need to add images first!" << std::endl;
    }
}
