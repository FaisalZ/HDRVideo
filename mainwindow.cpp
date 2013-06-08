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
    ui->scrollArea->hide();
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

void MainWindow::on_button_open_under_clicked()
{
    under_list = QFileDialog::getOpenFileNames(this,tr("Open underexposed Images"), ".",tr("Image Files (*.png)"));
    under_list.sort();
    if (under_list.length() > 0)
    {
        cv::Mat image= cv::imread(under_list[0].toLatin1().data(), -1);
        //image = cvLoadImage(under_list[0].toLatin1().data(),2);
        show_cvimg(image,0);
    }
}

void MainWindow::on_button_open_over_clicked()
{
    over_list = QFileDialog::getOpenFileNames(this,tr("Open overexposed Images"), ".",tr("Image Files (*.png)"));
    over_list.sort();
    if (over_list.length() > 0)
    {
        cv::Mat image= cv::imread(over_list[0].toLatin1().data(), -1);
        //image = cvLoadImage(over_list[0].toLatin1().data(),2);
        show_cvimg(image,1);
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
        int image_position = 0;


        //perform for every 10th frame
        while(image_position < over_list.size() && image_position < under_list.size())
        {
            //load two images
            cv::Mat under_img = cv::imread(under_list[image_position].toLatin1().data());
            cv::Mat over_img  = cv::imread(over_list[image_position].toLatin1().data());

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
            if(image_position == 0)
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
            if(image_position == 0)
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

            if(image_position == 0)
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
            if(image_position == 0)
            {
                ui->label_time->setNum((double)(((int)(100.0*(((double)cv::getTickCount() - t)/cv::getTickFrequency())))/100.0));
            }
            image_position = image_position+10;
        }//end while (every 10th frame)
        //create a csv file to store the Homography
        QFile file("/Users/FaisalZ/Desktop/out.csv");
        file.open(QIODevice::WriteOnly | QIODevice::Text);
        QTextStream out(&file);
        int counter = 0;
        Hom.at(0).copyTo(H);
        for(int i = 0; i < Hom.size(); i++)
        {
            counter++;
            if(i != 0)
            {
                //H = H+Hom.at(i);
            }
            /*cv::Mat p00(1,3,H.type());
            p00.at<double>(0,0) = 0;
            p00.at<double>(0,1) = 0;
            p00.at<double>(0,2) = 1;
            cv::Mat p01(1,3,H.type());
            p01.at<double>(0,0) = under_img_key.cols-1;
            p01.at<double>(0,1) = 0;
            p01.at<double>(0,2) = 1;
            cv::Mat p10(1,3,H.type());
            p10.at<double>(0,0) = 0;
            p10.at<double>(0,1) = under_img_key.rows-1;
            p10.at<double>(0,2) = 1;
            cv::Mat p11(1,3,H.type());
            p11.at<double>(0,0) = under_img_key.cols-1;
            p11.at<double>(0,1) = under_img_key.rows-1;
            p11.at<double>(0,2) = 1;
            cv::Mat p00x = H*p00;
            cv::Mat p01x = H*p01;
            cv::Mat p10x = H*p10;
            cv::Mat p11x = H*p11;
            out << QString::number(H.at<double>(0,0))+" "+QString::number(H.at<double>(1,0))+" "+QString::number(H.at<double>(2,0))+"\n";
            out << QString::number(H.at<double>(1,0))+" "+QString::number(H.at<double>(1,1))+" "+QString::number(H.at<double>(1,2))+"\n";
            out << QString::number(H.at<double>(2,0))+" "+QString::number(H.at<double>(2,1))+" "+QString::number(H.at<double>(2,2))+"\n\n";
            out << QString::number(p00x.at<double>(0,0))+" "+QString::number(p00x.at<double>(0,1))+" "+QString::number(p00x.at<double>(0,2))+"\n";
            out << QString::number(p01x.at<double>(0,0))+" "+QString::number(p01x.at<double>(0,1))+" "+QString::number(p01x.at<double>(0,2))+"\n";
            out << QString::number(p10x.at<double>(0,0))+" "+QString::number(p10x.at<double>(0,1))+" "+QString::number(p10x.at<double>(0,2))+"\n";
            out << QString::number(p11x.at<double>(0,0))+" "+QString::number(p11x.at<double>(0,1))+" "+QString::number(p11x.at<double>(0,2))+"\n";*/
            /*out << QString::number(i)+"\n"+QString::number(p01.at<double>(0,0)/p01.at<double>(3,0))+" "+QString::number(p01.at<double>(1,0)/p01.at<double>(3,0))+";\n";
            out << QString::number(i)+"\n"+QString::number(p10.at<double>(0,0)/p10.at<double>(3,0))+" "+QString::number(p10.at<double>(1,0)/p10.at<double>(3,0))+";\n";
            out << QString::number(i)+"\n"+QString::number(p11.at<double>(0,0)/p11.at<double>(3,0))+" "+QString::number(p11.at<double>(1,0)/p11.at<double>(3,0))+";\n";*/
        }
        //H.mul(1/counter);
        file.close();
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

        //obsolent: croping images to not view black edges
        /*cv::Mat p00(1,3,H.type());
        p00.at<double>(0,0) = 0;
        p00.at<double>(1,0) = 0;
        p00.at<double>(2,0) = 1;
        cv::Mat p01(1,3,H.type());
        p01.at<double>(0,0) = result.cols-1;
        p01.at<double>(1,0) = 0;
        p01.at<double>(2,0) = 1;
        cv::Mat p10(1,3,H.type());
        p10.at<double>(0,0) = 0;
        p10.at<double>(1,0) = result.rows-1;
        p10.at<double>(2,0) = 1;
        cv::Mat p11(1,3,H.type());
        p11.at<double>(0,0) = result.cols-1;
        p11.at<double>(1,0) = result.rows-1;
        p11.at<double>(2,0) = 1;
        p00 = p00*H;
        p01 = p01*H;
        p10 = p10*H;
        p11 = p11*H;
        int t,b,l,r;
        (p00.at<double>(1,0) > p01.at<double>(1,0)) ? t = p00.at<double>(1,0) : t = p01.at<double>(1,0);
        (p00.at<double>(0,0) > p10.at<double>(0,0)) ? l = p00.at<double>(0,0) : l = p10.at<double>(0,0);
        (p01.at<double>(1,0) < p11.at<double>(1,0)) ? r = p01.at<double>(1,0) : r = p11.at<double>(1,0);
        (p10.at<double>(0,0) < p11.at<double>(0,0)) ? b = p10.at<double>(0,0) : b = p11.at<double>(0,0);

        cv::Mat subImg = result(cv::Rect(l,t,r-l,b-t));
        result = subImg.resize(result.size,1);*/

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
            QMessageBox msgBox;
            msgBox.setText(QString::number(e->key()));
            msgBox.exec();
            }
        }
    }
}


/*Method that reads an png image and converts it into linear
 *@param    &path       path to image
 *@return   cv::Mat     linearized image
 */
cv::Mat MainWindow::read_linear(QString path)
{
    cv::Mat pic = cv::imread(path.toLatin1().data(), -1);
    QString identifer = path.right(31);
    int iso = 0;
    double lowerClip, upperClip,a,b,c,d,e,f;
    identifer = identifer.left(7);
    if(identifer == "001C003" ||
       identifer == "001C011" ||
       identifer == "005C003" ||
       identifer == "005C005")
    {
        iso = 1600;
        lowerClip = floor(0.125*65535);
        upperClip = ceil(0.915*65535);
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
        lowerClip = floor(0.15*65535);
        upperClip = ceil(0.882*65535);
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
        msgBox.setText("Unable to identify iso value.");
        msgBox.exec();
    }
    else
    {
        pic = cv::imread(path.toLatin1().data(), -1);
        double index = e*lowerClip+f;
        for(int y = 0; y < pic.rows; y++)
        {
            for(int x = 0; x < pic.cols; x++)
            {
                if(pic.at<double>(x,y) > upperClip)
                {
                    pic.at<double>(x,y) = upperClip;
                }
                if(pic.at<double>(x,y) > index)
                {
                    pic.at<double>(x,y) = (qPow(10,(pic.at<double>(x,y)-d)/c)-b)/a;
                }
                else
                {
                    pic.at<double>(x,y) = (pic.at<double>(x,y)-f)/e;
                }
            }
        }
    }
    return pic;
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




















