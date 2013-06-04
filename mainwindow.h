#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <QTextStream>
#include <QStringList>
#include <QLabel>
#include <QKeyEvent>

namespace Ui {
class MainWindow;
}

class MainWindow : public QMainWindow
{
    Q_OBJECT
    
public:
    explicit MainWindow(QWidget *parent = 0);
    ~MainWindow();
    
private slots:

    void show_cvimg(cv::Mat &img, int);

    void slider_sets_feature_value(int value);

    void slider_sets_nndr_value(int value);

    void slider_sets_nMatches_value(int value);

    void slider_sets_zoom_value(int value);

    void on_button_open_under_clicked();

    void on_button_open_over_clicked();

    void on_button_calc_offset_clicked();

    void on_button_apply_clicked();

    void keyPressEvent(QKeyEvent *e);

    cv::Mat read_linear(QString path);

private:
    Ui::MainWindow *ui;
    //Vectores storing the corresponding points
    std::vector< cv::Point2f > matchpoints_1,matchpoints_2;
    QStringList under_list, over_list;
    cv::Mat H;
    //custom adjustable
    double feature_tresh;
    int nMatches;
    //nearest neighbor distance ratio
    float nndRatio;
    //displayed images
    QLabel *img_left;
    QLabel *img_right;
    QLabel *img_bottom;
    //position for review
    int position;
    bool ready;
};

#endif // MAINWINDOW_H
