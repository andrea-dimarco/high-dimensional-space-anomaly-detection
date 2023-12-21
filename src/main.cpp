// Main

#include "include/suv.h"
#include "include/gem.h"
#include "include/pca.h"
#include <iostream>
#include <assert.h>
#include <random>
#include <cstring>
#include <cstdlib>
#include <assert.h>
#include <ctime>

#include <Eigen/Dense>
//#include <eigen3/Eigen/Dense>

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
    int p = 3; // output dimension
    // sensors are not independent within eachother at time t
    // different samples taken ad different times t and t' are i.i.d.

    int tau = 0; // change-point (when the anomaly begins)

    // the model is unkown so must be simulated as i.i.d. variables

    int N = 100; // number of samples in the nominal data set (data guaranteed to have no anomalies)

    // testing area
    std::clock_t start;
    double duration;
    start = std::clock();

    Eigen::MatrixXd X = random_dataset(p, N, false/*normal*/); // nominal dataset

    SUV suv;
     
    //GEM gem(p);

    PCA pca(p);
    std::cout << "Samples loaded" << std::endl;

    pca.offline_phase(X);
    std::cout << "Finished PCA offline phase" << std::endl;

    // // X = suv.open_data("datasets/nominal-human-activity.csv");
    // // p = X.rows(); N = X.cols();
    // // GEM gem(p);
    // // std::cout << "Nominal samples loaded!!" << std::endl << "Dimension: " << p << std::endl << "Samples: " << N << std::endl;

    // gem.offline_phase(X);
    // // duration = ( std::clock() - start ) / (double) CLOCKS_PER_SEC;
    // // std::cout<<"Duration: "<< duration << "s" << std::endl;
    // std::cout << "Offline phase done!!" << std::endl;

    // // // GEM Online phase
    // X = random_dataset(p, N, true/*uniform*/, -1.0, 0.0); // anomalous dataset
    // // X = suv.open_data("datasets/anomaly-human-activity.csv"); // anomaly!!
    // p = X.rows(); N = X.cols();
    // std::cout << "Anomalous samples loaded!!" << std::endl << "Dimension: " << p << std::endl << "Samples: " << N << std::endl;

    // std::cout << "Begin online phase..." << std::endl;
    // gem.load_model();
    // for (int i = 0; i < N; i++) {
    //     // anomaly detection
    //     if (gem.online_detection(X.col(i))) {
    //         std::cout << "Anomaly found with delay: " << (i-tau) << "!!" << std::endl;
    //         return 0;
    //     }
    // }
    // std::cout << "No anomaly found." << std::endl;
    return 0;
} /* main */
