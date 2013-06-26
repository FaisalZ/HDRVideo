#-------------------------------------------------
#
# Project created by QtCreator 2012-11-18T22:08:50
#
#-------------------------------------------------

QT       += core gui

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

TARGET = OpenCV_Matcher
TEMPLATE = app


SOURCES += mainwindow.cpp\
    main.cpp \
    linear.cpp \
    img_ops.cpp

HEADERS  += mainwindow.h \
    linear.h \
    img_ops.h

FORMS    += mainwindow.ui

INCLUDEPATH += $$PWD/../../../../opt/local/include/

macx: LIBS += -L$$PWD/../../../../opt/local/lib/ -lopencv_highgui
macx: LIBS += -L$$PWD/../../../../opt/local/lib/ -lopencv_core
macx: LIBS += -L$$PWD/../../../../opt/local/lib/ -lopencv_imgproc
macx: LIBS += -L$$PWD/../../../../opt/local/lib/ -lopencv_features2D
macx: LIBS += -L$$PWD/../../../../opt/local/lib/ -lopencv_nonfree
macx: LIBS += -L$$PWD/../../../../opt/local/lib/ -lopencv_calib3d
