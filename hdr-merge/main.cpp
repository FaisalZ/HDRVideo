#include <QCoreApplication>
#include <iostream>
#include <QStringList>
#include <controller.h>

int main(int argc, char *argv[])
{
    QCoreApplication a(argc, argv);
    double time;
    QStringList arguments = a.arguments();
    if(arguments.size() < 2)
    {
        arguments.append("-h");
    }
    time = time + controller::start(arguments);

    std::cout << std::endl;
    std::cout << "finished in " << time << "seconds." << std::endl;
    a.exit(0);
}
