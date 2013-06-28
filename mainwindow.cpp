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
#include <QMessageBox>
#include <QScrollArea>
#include <QKeyEvent>
#include <QProgressDialog>
#include <linear.h>
#include <img_ops.h>

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);
    connect( ui->slider_feature_tresh, SIGNAL( valueChanged(int) ), this   , SLOT(  slider_sets_feature_value(int) ) );
    connect( ui->slider_nndr, SIGNAL( valueChanged(int) ), this   , SLOT( slider_sets_nndr_value(int) ) );
    connect( ui->slider_nMatches, SIGNAL( valueChanged(int) ), this  , SLOT( slider_sets_nMatches_value(int) ) );
    connect( ui->slider_lower, SIGNAL( valueChanged(int) ), this  , SLOT( slider_sets_lower_value(int) ) );
    connect( ui->slider_upper, SIGNAL( valueChanged(int) ), this  , SLOT( slider_sets_upper_value(int) ) );
    connect( ui->slider_hdr_shift, SIGNAL( valueChanged(int) ), this  , SLOT( slider_sets_hdr_value(int) ) );
    //set initial values
    slider_sets_feature_value(ui->slider_feature_tresh->value());
    slider_sets_nndr_value(ui->slider_nndr->value());
    slider_sets_nMatches_value(ui->slider_nMatches->value());
    //slider_sets_lower_value(ui->slider_lower->value());
    //slider_sets_upper_value(ui->slider_upper->value());
    ui->label->hide();
    ui->scrollArea->hide();
    ui->button_join->setEnabled(true);
    ui->button_render->setEnabled(true);
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

void MainWindow::slider_sets_lower_value(int value)
{
    ui->label_lower->setNum((float)value/63.75f);
    img = img_ops::blend_images(img_u,img_o,((float)ui->slider_upper->value()/63.75f),((float)value/63.75f));
    slider_sets_hdr_value(ui->slider_hdr_shift->value());
}

void MainWindow::slider_sets_upper_value(int value)
{
    ui->label_upper->setNum((float)value/63.75f);
    img = img_ops::blend_images(img_u,img_o,((float)value/63.75f),((float)ui->slider_lower->value()/63.75f));
    slider_sets_hdr_value(ui->slider_hdr_shift->value());
}

void MainWindow::slider_sets_hdr_value(int value)
{
    float factor = qPow(2,((float)value/16.0f)-7);
    cv::Mat img_new = img*factor;
    img_new.convertTo(img_new, CV_8U, 256);
    show_cvimg(img_new,3);
}



void MainWindow::on_button_open_under_clicked()
{
    under_list = QFileDialog::getOpenFileNames(this,tr("Open underexposed Images"), ".",tr("Image Files (*.png)"));
    under_list.sort();
    if (under_list.length() > 0)
    {
        img_u = Linear::read_linear(under_list[0],1,min_u,max_u);
        show_cvimg(img_u,0);
    }
}

void MainWindow::on_button_open_over_clicked()
{
    over_list = QFileDialog::getOpenFileNames(this,tr("Open overexposed Images"), ".",tr("Image Files (*.png)"));
    over_list.sort();
    if (over_list.length() > 0)
    {
        img_o = Linear::read_linear(over_list[0],10,min_o,max_o);
        //image = cvLoadImage(over_list[0].toLatin1().data(),2);
        show_cvimg(img_o,1);
    }
}

void MainWindow::on_button_calc_offset_clicked()
{
    ui->label_under->show();
    ui->label_over->show();
    if(over_list.length() > 0 && under_list.length() > 0)
    {
        //start timer
        double t = (double)cv::getTickCount();

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

        //loop variables
        QVector< cv::Mat > Hom;

        //perform for every 10th frame
        int max;
        bool stop = false;
        (over_list.size() < under_list.size()) ? max = over_list.size() : max = under_list.size();
        QProgressDialog progress("Calculating Homography...", "Abort", 0, max);
                progress.setWindowModality(Qt::WindowModal);

        for(int looper = 0; looper < max; looper= looper+10)
        {
            progress.setValue(looper);
            if (progress.wasCanceled())
            {
                stop = true;
                break;
            }

            //load two images
            /*cv::Mat under_img = cv::imread(under_list[looper].toLatin1().data());
            cv::Mat over_img  = cv::imread(over_list[looper].toLatin1().data());*/
            cv::Mat under_img = Linear::read_linear(under_list[looper],1,min_o,max_o);
            cv::pow(under_img,(1.0f/2.2f),under_img);
            under_img.convertTo(under_img, CV_8U, 256);
            ui->label_2->setNum(under_img.type());
            cv::Mat over_img  = Linear::read_linear(over_list[looper],10,min_o,max_o);
            cv::pow(over_img,(1.0f/2.2f),over_img);
            over_img.convertTo(over_img, CV_8U, 256);

            //clear data
            keypoints_over.clear();
            keypoints_under.clear();
            good_matches.clear();

            //detect features
            cv::SiftFeatureDetector sifter(feature_tresh);
            sifter.detect(under_img,keypoints_under);
            sifter.detect(over_img,keypoints_over);

            //draw keypoints on image & display images
            cv::drawKeypoints(under_img,keypoints_under,under_img_key,cv::Scalar(0,255,0),cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
            cv::drawKeypoints(over_img,keypoints_over,over_img_key,cv::Scalar(0,255,0),cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
            if(looper == 0)
            {
                show_cvimg(under_img_key,0);
                show_cvimg(over_img_key,1);
                ui->label_features->setNum((int)keypoints_over.size());
            }

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
            if(looper == 0)
            {
                ui->label_matches->setNum((int)good_matches.size());
            }

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

            if(looper == 0)
            {
                //print Matches on images and show images
                cv::drawMatches(over_img,keypoints_over, // 1st image and its keypoints
                                under_img,keypoints_under, // 2nd image and its keypoints
                                show_matches,            // the matches
                                imageMatches,      // the image produced
                                cv::Scalar(255,255,255)); // color of the lines
                show_cvimg(imageMatches,2);
            }

            //get corresponding points
            for( int i = 0; i < good_matches.size(); i++ )
            {
                //-- Get the keypoints from the good matches
                matchpoints_1.push_back( keypoints_over[ good_matches[i].queryIdx ].pt );
                matchpoints_2.push_back( keypoints_under[ good_matches[i].trainIdx ].pt );
            }

            //calculate perspective transformation
            Hom.append(cv::findHomography(matchpoints_1,matchpoints_2,CV_RANSAC));
            //output time taken in seconds
            if(looper == 0)
            {
                ui->label_time->setNum((double)(((int)(100.0*(((double)cv::getTickCount() - t)/cv::getTickFrequency())))/100.0));
            }
        }//end for (every 10th frame)
        if (stop != true)
        {
            //create a csv file to store the Homography
            QString d = QFileInfo(over_list[0]).absoluteDir().absolutePath();
            QFile file(d+"/out.csv");
            file.open(QIODevice::WriteOnly | QIODevice::Text);
            QTextStream out(&file);
            int counter = 0;
            Hom.at(0).copyTo(H);
            for(int i = 0; i < Hom.size(); i++)
            {
                counter++;
                if(i != 0)
                {
                    add(H,Hom.at(i),H);
                    H=H/2;
                }
                cv::Mat p00(3,1,H.type());
                p00.at<double>(0,0) = 0;
                p00.at<double>(1,0) = 0;
                p00.at<double>(2,0) = 1;
                cv::Mat p01(3,1,H.type());
                p01.at<double>(0,0) = under_img_key.cols-1;
                p01.at<double>(1,0) = 0;
                p01.at<double>(2,0) = 1;
                cv::Mat p10(3,1,H.type());
                p10.at<double>(3,1) = 0;
                p10.at<double>(1,0) = under_img_key.rows-1;
                p10.at<double>(2,0) = 1;
                cv::Mat p11(3,1,H.type());
                p11.at<double>(0,0) = under_img_key.cols-1;
                p11.at<double>(1,0) = under_img_key.rows-1;
                p11.at<double>(2,0) = 1;
                cv::Mat p00x;
                gemm(Hom.at(i),p00,1,NULL,0,p00x);
                cv::Mat p01x;
                gemm(Hom.at(i),p01,1,NULL,0,p01x);
                cv::Mat p10x;
                gemm(Hom.at(i),p10,1,NULL,0,p10x);
                cv::Mat p11x;
                gemm(Hom.at(i),p11,1,NULL,0,p11x);
                /*if (i == 0)
                {
                    //Output Homography Matrix
                    out << QString::number(Hom.at(i).at<double>(0,0))+" "+QString::number(Hom.at(i).at<double>(0,1))+" "+QString::number(Hom.at(i).at<double>(0,2))+"\n";
                    out << QString::number(Hom.at(i).at<double>(0,1))+" "+QString::number(Hom.at(i).at<double>(1,1))+" "+QString::number(Hom.at(i).at<double>(2,1))+"\n";
                    out << QString::number(Hom.at(i).at<double>(0,2))+" "+QString::number(Hom.at(i).at<double>(1,2))+" "+QString::number(Hom.at(i).at<double>(2,2))+"\n\n";
                }*/
                //Output Points
                out << QString::number(p00x.at<double>(0,0)/p00x.at<double>(2,0))+","+QString::number(p00x.at<double>(1,0)/p00x.at<double>(2,0))+",";
                out << QString::number(p01x.at<double>(0,0)/p01x.at<double>(2,0))+","+QString::number(p01x.at<double>(1,0)/p01x.at<double>(2,0))+",";
                out << QString::number(p10x.at<double>(0,0)/p10x.at<double>(2,0))+","+QString::number(p10x.at<double>(1,0)/p10x.at<double>(2,0))+",";
                out << QString::number(p11x.at<double>(0,0)/p11x.at<double>(2,0))+","+QString::number(p11x.at<double>(1,0)/p11x.at<double>(2,0))+"\n";
                /*out << QString::number(i)+"\n"+QString::number(p01.at<double>(0,0)/p01.at<double>(3,0))+" "+QString::number(p01.at<double>(1,0)/p01.at<double>(3,0))+";\n";
                out << QString::number(i)+"\n"+QString::number(p10.at<double>(0,0)/p10.at<double>(3,0))+" "+QString::number(p10.at<double>(1,0)/p10.at<double>(3,0))+";\n";
                out << QString::number(i)+"\n"+QString::number(p11.at<double>(0,0)/p11.at<double>(3,0))+" "+QString::number(p11.at<double>(1,0)/p11.at<double>(3,0))+";\n";*/
            }
            file.close();
            ui->button_join->setEnabled(true);
            ui->button_render->setEnabled(true);
        }
        else //progress aborted
        {
            H.release();
            Hom.clear();
        }
    }
    else
    {
        QMessageBox msgBox;
        msgBox.setText("You need to add images first!");
        msgBox.exec();
    }
}

void MainWindow::on_button_apply_clicked()
{
    if(over_list.length() > 0 && under_list.length() > 0 && !H.empty())
    {
        cv::Mat over_img = cv::imread(over_list[0].toLatin1().data(), -1);
        cv::Mat under_img = cv::imread(under_list[0].toLatin1().data(), -1);
        cv::Mat result;
        cv::warpPerspective(over_img,result,H,cv::Size(under_img.cols,under_img.rows));

        show_cvimg(result,3);
        position = 0;
        ready = true;
        QString filename = QFileDialog::getSaveFileName( this,
                                                         tr("Save Warped Image"),
                                                         QDir::homePath(),
                                                         tr("Documents (*.png)") );
        if(filename != "")
        {
            cv::imwrite(filename.toLatin1().data(),result);
        }
    }
    else
    {
        QMessageBox msgBox;
        msgBox.setText("You need to calculate the offset first!");
        msgBox.exec();
    }
}

void MainWindow::on_button_join_clicked()
{
    if(over_list.length() > 0 && under_list.length() > 0 && !H.empty() && !img_o.empty() && !img_u.empty())
    {
        cv::warpPerspective(img_o,img_o,H,cv::Size(img_o.cols,img_o.rows));

        int upper = ui->slider_upper->value();
        int lower = ui->slider_lower->value();

        img = img_ops::blend_images(img_u,img_o,((float)upper/63.75f),((float)lower/63.75f));

        slider_sets_hdr_value(ui->slider_hdr_shift->value());
        ui->slider_hdr_shift->setEnabled(true);
        ui->slider_upper->setEnabled(true);
        ui->slider_lower->setEnabled(true);
    }
    else
    {
        QMessageBox msgBox;
        msgBox.setText("No offest calculated!");
        msgBox.exec();

        int upper = ui->slider_upper->value();
        int lower = ui->slider_lower->value();

        img = img_ops::blend_images(img_u,img_o,((float)upper/63.75f),((float)lower/63.75f));

        slider_sets_hdr_value(ui->slider_hdr_shift->value());
        ui->slider_hdr_shift->setEnabled(true);
        ui->slider_upper->setEnabled(true);
        ui->slider_lower->setEnabled(true);
    }
}

void MainWindow::keyPressEvent(QKeyEvent *e)
{
    if(ready)
    {
        switch( e->key())
        {
            case 87: //up(w)
            {
                cv::Mat img = cv::imread(under_list[position].toLatin1().data(), -1);
                show_cvimg(img,3);
                /*QMessageBox msgBox;
                msgBox.setText("Press ↑");
                msgBox.exec();*/
                break;
            }
            case 83: //down(s)
            {
                cv::Mat over_img = cv::imread(over_list[position].toLatin1().data(), -1);
                cv::Mat under_img = cv::imread(under_list[position].toLatin1().data(), -1);
                cv::Mat img;
                cv::warpPerspective(over_img,img,H,cv::Size(under_img.cols,under_img.rows));
                show_cvimg(img,3);
                break;
            }
            case 65: //left(a)
            {
                if(position-1 >= 0)
                {
                    position--;
                    cv::Mat img = cv::imread(under_list[position].toLatin1().data(), -1);
                    show_cvimg(img,3);
                }
                break;
            }
            case 68: //right(d)
            {
            if(position+1 < over_list.length() && position+1 < under_list.length())
            {
                position++;
                cv::Mat img = cv::imread(under_list[position].toLatin1().data(), -1);
                show_cvimg(img,3);
            }
                break;
            }
            default:
            {
            /*QMessageBox msgBox;
            msgBox.setText(QString::number(e->key()));
            msgBox.exec();*/
            }
        }
    }
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
    //convert img from 16bit to 8bit
    if (img_new.type() == 18)
    {
        img_new.convertTo(img_new, CV_8U, 0.00390625);
    }
    //convert from float to 8bit
    if (img_new.type() == 21)
    {
        cv::pow(img_new,(1.0f/2.2f),img_new);
        img_new.convertTo(img_new, CV_8U, 256);
    }
    // convert cv::Mat to QImage and resize the image to fit the lable
    QImage qimg= QImage((const unsigned char*)(img_new.data),img_new.cols,img_new.rows,QImage::Format_RGB888);
    // display QImage on label
    if (f == 0)
    {
        ui->label_under->show();
        ui->label_over->show();
        ui->label->hide();
        ui->scrollArea->hide();
        qimg = qimg.scaledToWidth(500);
        ui->label_under->setPixmap(QPixmap::fromImage(qimg));
        ui->label_under->resize(ui->label_under->pixmap()->size());
    }
    else if (f == 1)
    {
        ui->label_under->show();
        ui->label_over->show();
        ui->label->hide();
        ui->scrollArea->hide();
        qimg = qimg.scaledToWidth(500);
        ui->label_over->setPixmap(QPixmap::fromImage(qimg));
        ui->label_over->resize(ui->label_over->pixmap()->size());
    }
    else if (f == 2)
    {
        ui->label->show();
        qimg = qimg.scaledToWidth(1005);
        ui->label->setPixmap(QPixmap::fromImage(qimg));
        ui->label->resize(ui->label->pixmap()->size());
    }
    else if (f == 3)
    {
        int v = ui->scrollArea->verticalScrollBar()->value();
        int h = ui->scrollArea->horizontalScrollBar()->value();
        ui->label_under->hide();
        ui->label_over->hide();
        ui->label->hide();
        ui->scrollArea->show();
        img_right = new QLabel;
        img_right->setPixmap(QPixmap::fromImage(qimg));
        ui->scrollArea->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
        ui->scrollArea->setBackgroundRole(QPalette::Dark);
        ui->scrollArea->setWidget(img_right);
        ui->scrollArea->verticalScrollBar()->setValue(v);
        ui->scrollArea->horizontalScrollBar()->setValue(h);
    }
}

void MainWindow::on_button_render_clicked()
{
    int max;
    (over_list.size() < under_list.size()) ? max = over_list.size() : max = under_list.size();
    for(int i = 0; i < max; i++)
    {
        cv::Mat over = Linear::read_linear(over_list[i],10,min_u,max_u);
        cv::Mat under = Linear::read_linear(under_list[i],1,min_u,max_u);

        cv::warpPerspective(over,over,H,cv::Size(under.cols,under.rows));

        int upper = ui->slider_upper->value();
        int lower = ui->slider_lower->value();
        cv::Mat linear = img_ops::blend_images(under,over,((float)upper/63.75f),((float)lower/63.75f));
        cv::Mat loga;
        cv::log((((linear*65535.0)+(qPow(2.0,-16.0)))+16),loga);
        loga = loga*(65535.0/32.0);
        loga.convertTo(loga,18,1);

        QString linear_filename;
        QString log_filename;

        QString path = QFileInfo(over_list[0]).absoluteDir().absolutePath();
        linear_filename = path+"/"+QFileInfo(over_list[0]).completeBaseName().right(27)+"_lin.exr";
        log_filename = path+"/"+QFileInfo(over_list[0]).completeBaseName().right(27)+"_log.png";

        cv::imwrite(linear_filename.toLatin1().data(),linear);
        cv::imwrite(log_filename.toLatin1().data(),loga);
    }
}
