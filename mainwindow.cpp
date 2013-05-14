#include "mainwindow.h"
#include "ui_mainwindow.h"
#include "opencv2/opencv.hpp"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/flann/kdtree_index.h>
#include <opencv2/stitching/detail/matchers.hpp>
#include <QTextStream>
#include <QFileDialog>
#include <QTime>
#include <qmath.h>
#include <QLabel>

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);
    connect( ui->slider_feature_tresh, SIGNAL( valueChanged(int) ), this   , SLOT(  slider_sets_feature_value(int) ) );
    connect( ui->slider_nndr, SIGNAL( valueChanged(int) ), this   , SLOT( slider_sets_nndr_value(int) ) );
    connect( ui->slider_nMatches, SIGNAL( valueChanged(int) ), this  , SLOT( slider_sets_nMatches_value(int) ) );
    //set initial values
    slider_sets_feature_value(ui->slider_feature_tresh->value());
    slider_sets_nndr_value(ui->slider_nndr->value());
    slider_sets_nMatches_value(ui->slider_nMatches->value());
    ui->label->hide();
}

MainWindow::~MainWindow()
{
    delete ui;
}

void MainWindow::slider_sets_feature_value(int value)
{
    ui->label_feature_tresh->setNum((double)value/100);
    feature_tresh = (double)value/100;
}

void MainWindow::slider_sets_nndr_value(int value)
{
    ui->label_nndr->setNum((double)value/100);
    nndRatio = (double)value/100;
}

void MainWindow::slider_sets_nMatches_value(int value)
{
    ui->label_nMatches->setNum(value);
    nMatches = value;
}


/*Method that converts cv::Mat to QImage and displays it on a label
 *@param    &img    pointer to cv::Mat
 *@param    f       index of target label (0 -> left, 1 -> right)
 */
void MainWindow::show_cvimg(cv::Mat &img, int f)
{
    // change color channel ordering from BGR to RGB
    cv::Mat img_new;
    cv::cvtColor(img,img_new,CV_BGR2RGB);
    //double shift = qPow(2,4);
    //cv::convertScaleAbs(img_new,img_new,0.25,shift);
    // convert cv::Mat to QImage and resize the image to fit the lable
    QImage qimg= QImage((const unsigned char*)(img_new.data),img_new.cols,img_new.rows,QImage::Format_RGB888);
    // display QImage on label
    if (f == 0)
    {
        qimg = qimg.scaledToWidth(500);
        ui->img_under->setPixmap(QPixmap::fromImage(qimg));
        ui->img_under->resize(ui->img_under->pixmap()->size());
    }
    else if (f == 1)
    {
        qimg = qimg.scaledToWidth(500);
        ui->img_over->setPixmap(QPixmap::fromImage(qimg));
        ui->img_over->resize(ui->img_over->pixmap()->size());
    }
    else if (f == 2)
    {
        qimg = qimg.scaledToWidth(1005);
        ui->label->setPixmap(QPixmap::fromImage(qimg));
        ui->label->resize(ui->label->pixmap()->size());
    }
}

void MainWindow::on_button_open_under_clicked()
{
    under_list = QFileDialog::getOpenFileNames(this,tr("Open underexposed Images"), ".",tr("Image Files (*.png)"));
    if (under_list.length() > 0)
    {
        image= cv::imread(under_list[0].toLatin1().data());
        //image = cvLoadImage(under_list[0].toLatin1().data(),2);
        show_cvimg(image,0);
    }
}

void MainWindow::on_button_open_over_clicked()
{
    over_list = QFileDialog::getOpenFileNames(this,tr("Open overexposed Images"), ".",tr("Image Files (*.png)"));
    if (under_list.length() > 0)
    {
        image= cv::imread(over_list[0].toLatin1().data());
        //image = cvLoadImage(over_list[0].toLatin1().data(),2);
        show_cvimg(image,1);
    }
}

void MainWindow::on_button_calc_offset_clicked()
{
    double t = (double)cv::getTickCount();
    //load two images
    under_img = cv::imread(under_list[0].toLatin1().data());
    over_img  = cv::imread(over_list[0].toLatin1().data());

    //initialise needed Variables
        //keypoints
        std::vector< cv::KeyPoint > keypoints_under;
        std::vector< cv::KeyPoint > keypoints_over;
        //Keypoint Descriptors
        cv::Mat desc_under,desc_over;
        //vectors holding the two best nearest neighbors per descriptor
        std::vector< std::vector< cv::DMatch > > matches1;
        std::vector< std::vector< cv::DMatch > > matches2;
        //images  with keypoints on top
        cv::Mat under_img_key,over_img_key;
        //vector holding the best nearest neighbor per descriptor
        std::vector< cv::DMatch > good_matches;
        //image showing matches (connected with lines)
        cv::Mat imageMatches;

    //detect features
    cv::SiftFeatureDetector sifter(feature_tresh);
    sifter.detect(under_img,keypoints_under);
    sifter.detect(over_img,keypoints_over);

    //draw keypoints on image & display images
    cv::drawKeypoints(under_img,keypoints_under,under_img_key,cv::Scalar(0,255,0),cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    cv::drawKeypoints(over_img,keypoints_over,over_img_key,cv::Scalar(0,255,0),cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    show_cvimg(under_img_key,0);
    show_cvimg(over_img_key,1);
    ui->label_features->setNum((int)keypoints_over.size());

    //run Descriptor Extractor in keypoints
    cv::SiftDescriptorExtractor siftDesc;
    siftDesc.compute(under_img,keypoints_under,desc_under);
    siftDesc.compute(over_img,keypoints_over,desc_over);

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
            if(matches1[i][0].distance/matches1[i][1].distance > nndRatio)
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
            if(matches2[i][0].distance/matches2[i][1].distance > nndRatio)
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
    ui->label_matches->setNum((int)good_matches.size());

    //create a vector of Matches to be shown
    std::vector< cv::DMatch > show_matches(good_matches.size());
    std::copy(good_matches.begin(),good_matches.end(),show_matches.begin());
    if (show_matches.size() > nMatches)
    {
        std::nth_element(   show_matches.begin(),
                            show_matches.begin()+nMatches,
                            show_matches.end());
        show_matches.erase(show_matches.begin()+nMatches+1, show_matches.end());
    }
    //print Matches on images and show images
    cv::drawMatches(over_img,keypoints_over, // 1st image and its keypoints
                    under_img,keypoints_under, // 2nd image and its keypoints
                    show_matches,            // the matches
                    imageMatches,      // the image produced
                    cv::Scalar(255,255,255)); // color of the lines
    ui->label->show();
    show_cvimg(imageMatches,2);

    //get corresponding points
    for( int i = 0; i < good_matches.size(); i++ )
    {
        //-- Get the keypoints from the good matches
        matchpoints_1.push_back( keypoints_over[ good_matches[i].queryIdx ].pt );
        matchpoints_2.push_back( keypoints_under[ good_matches[i].trainIdx ].pt );
    }
    cv::Mat H = cv::findHomography(matchpoints_1,matchpoints_2,CV_RANSAC);
    cv::Mat result;
    cv::warpPerspective(over_img,result,H,cv::Size(under_img.cols,under_img.rows));
    cv::imwrite("/Users/FaisalZ/Desktop/test.png",result);
    ui->label_time->setNum((double)(((int)(100.0*(((double)cv::getTickCount() - t)/cv::getTickFrequency())))/100.0));
}

void MainWindow::on_button_apply_clicked()
{
    ui->img_over->hide();
    ui->label->hide();
    //show_cvimg(result,0);
}
