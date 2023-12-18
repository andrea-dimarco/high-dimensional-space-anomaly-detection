// Main

#include "include/suv.h"
#include "include/gem.h"

#include <iostream>
#include <assert.h>
#include <random>
#include <cstring>
#include <cstdlib>
#include <assert.h>

#include <Eigen/Dense>

/**
 * Generate a matrix of random values given 5 parameters
 * Dimension on x axis,
 * Dimension on y axis,
 * is_uniform use a uniform distribution (otherwisen it will be Gaussian)
 * lower_bound (or mean)
 * upper_bound (or standard deviation)
 * seed (the seed for the rasndom generator)
*/
Eigen::MatrixXd random_dataset(int dim0=2, int dim1=2, bool is_uniform=true, double a=0.0, double b=1.0, int seed=0) {
    
    assert((dim0>0) && (dim1>0));

    int i,j;
    std::default_random_engine prg(seed);
    Eigen::MatrixXd dataset;
    dataset.resize(dim0, dim1);

    if(is_uniform) {
        std::uniform_real_distribution<double> uniform(a, b);
        for (i = 0; i < dim0; i++) {
            for (j = 0; j < dim1; j ++) {
                dataset(i,j) = uniform(prg);
            }
        }
    } else { // is gaussian
        std::normal_distribution normal(a,b);
        for (i = 0; i < dim0; i++) {
            for (j = 0; j < dim1; j ++) {
                dataset(i,j) = normal(prg);
            }
        }
    }
    return dataset;
} /* random_dataset */

/**
 * Main function of our SUV.
 */
int main()
{
    int p = 10; // output dimension
    // sensors are not independent within eachother at time t
    // different samples taken ad different times t and t' are i.i.d.

    int tau = 10; // change-point (when the anomaly begins)

    // the model is unkown so must be simulated as i.i.d. variables

    int N = 100; // number of samples in the nominal data set (data guaranteed to have no anomalies)

    // testing area 
    Eigen::MatrixXd X = random_dataset(p, N, false/*normal*/); // nominal dataset
    Eigen::MatrixXd Z = random_dataset(p, N, false/*normal*/, 0.5, 1.0); // poisoned dataset

    for (int i = 0; i < tau; i++) {
        Z.col(i) = X.col(i);
    }



    GEM gem(p);
    gem.offline_phase(X);

    for (int i = 0; i < N; i++) {
        if (gem.online_detection(Z.col(i))) {
            std::cout << "Anomaly found with delay: " << (i-tau) << "!!" << std::endl;
            return 0;
        }
    }
    std::cout << "No anomaly found." << std::endl;
    return 0;
} /* main */
