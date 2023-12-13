// Main

#include <iostream>
#include <random>
#include <cstring>
#include <cstdlib>

using namespace std;

double mean = 0.0;
double stddev = 1.0;
default_random_engine gen;
uniform_real_distribution<double> dist(mean, stddev);

class Point {
    private:
        double x;
        double y;
        int    id;

    public:
        Point(double xp=0, double yp=0, bool random_init=true) {
            if (random_init) {
                x = (double)dist(gen);
                y = (double)dist(gen);
            } else {
                x = xp;
                y = yp;
            }
            id = abs((int)(10*dist(gen)));
        }
        void print_point() {
            cout << id << ": (" << x << ", " << y << ")" << endl;
        }
        void get_coordinates(double coordinates[]) {
            coordinates[0] = x;
            coordinates[1] = y;
        }
};

/**
 * Main function of our SUV.
 */
int main()
{

    // PRG
    gen.seed(0);


    // Test
    cout << "Random number Whazaa!: " << dist(gen) << endl;

    Point p0(0.0, 0.0, false);
    p0.print_point();

    Point p1;
    p1.print_point();
    
    return 0;
} /* main */
