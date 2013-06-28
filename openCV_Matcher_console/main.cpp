#include <QCoreApplication>
#include <iostream>
#include <QStringList>
#include <controller.h>

int main(int argc, char *argv[])
{
    QCoreApplication a(argc, argv);
    double time;
    QStringList arguments = a.arguments();
    std::cout << "Found " <<arguments.length()-1 << " scenes to be rendered!" <<std::endl;
    std::cout << std::endl;

    for(int i = 1; i < arguments.length(); i++)
    {
        std::cout << "------------ Starting scene "<< i << " ------------"<< std::endl;
        time = time + controller::start(arguments[i]);
    }

    std::cout << std::endl;
    std::cout << "Seconds needed in total: " << time << std::endl;
    std::cout << "\\\\\\\\\\\\ ALL DONE ////////////" << std::endl;
    a.exit(0);
    //return a.exec();
}
