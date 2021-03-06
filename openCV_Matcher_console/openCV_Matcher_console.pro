#-------------------------------------------------
#
# Project created by QtCreator 2013-06-25T21:48:07
#
#-------------------------------------------------

QT       += core

QT       -= gui

TARGET = openCV_Matcher_console
CONFIG   += console
CONFIG   -= app_bundle

TEMPLATE = app


SOURCES += main.cpp \
    matcher.cpp \
    controller.cpp \
    fileio.cpp \
    renderer.cpp \
    img_ops.cpp

HEADERS += \
    matcher.h \
    controller.h \
    fileio.h \
    renderer.h \
    img_ops.h

INCLUDEPATH += $$PWD/../../../../opt/local/include/

macx: LIBS += -L$$PWD/../../../../opt/local/lib/ -lopencv_highgui
macx: LIBS += -L$$PWD/../../../../opt/local/lib/ -lopencv_core
macx: LIBS += -L$$PWD/../../../../opt/local/lib/ -lopencv_imgproc
macx: LIBS += -L$$PWD/../../../../opt/local/lib/ -lopencv_features2D
macx: LIBS += -L$$PWD/../../../../opt/local/lib/ -lopencv_nonfree
macx: LIBS += -L$$PWD/../../../../opt/local/lib/ -lopencv_calib3d
