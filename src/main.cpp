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

//#include <python.h>

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

    // Py_Initialize();
    // PyRun_SimpleString("print('hello world')");
    // Py_Finalize();
    // return 0;
    int p = 10; // output dimension
    // sensors are not independent within eachother at time t
    // different samples taken ad different times t and t' are i.i.d.

    int tau = 0; // change-point (when the anomaly begins)

    // the model is unkown so must be simulated as i.i.d. variables

    int N = 1000; // number of samples in the nominal data set (data guaranteed to have no anomalies)

    // testing area
    std::clock_t start;
    double duration;
    start = std::clock();

    Eigen::MatrixXd X = random_dataset(p, N, false/*normal*/); // nominal dataset

    SUV suv;

    PCA model(p);
    std::cout << "Samples loaded" << std::endl;

    //model.offline_phase(X);
    std::cout << "Finished PCA offline phase" << std::endl;

    // // X = suv.open_data("datasets/nominal-human-activity.csv");
    // // p = X.rows(); N = X.cols();
    // // std::cout << "Nominal samples loaded!!" << std::endl << "Dimension: " << p << std::endl << "Samples: " << N << std::endl;

    //model.offline_phase(X);
    duration = ( std::clock() - start ) / (double) CLOCKS_PER_SEC;
    std::cout << "Duration: " << duration << "s" << std::endl;
    std::cout << "Offline phase done!!" << std::endl;

    // // // GEM Online phase
    //X = random_dataset(p, N, false/*normal*/); // anomalous dataset
    // X = suv.open_data("datasets/anomaly-human-activity.csv"); // anomaly!!
    p = X.rows(); N = X.cols();
    std::cout << "Anomalous samples loaded!!" << std::endl << "Dimension: " << p << std::endl << "Samples: " << N << std::endl;

    std::cout << "Begin online phase..." << std::endl;
    //std::cout << X.col(539) << std::endl;
    std::cout << X.col(95) << std::endl; // <- With PCA this goes to inf and anomaly is detected p=10, N=1000;
    model.load_model();
    double max_confidence = 0.0;
    int anomaly_count = 0;
    for (int i = 94; i < N; i++) {
        // std::cout << "Current confidence: " << model.get_g() << std::endl;
        if (model.get_g() > max_confidence) {
            max_confidence = model.get_g();
        }
        // anomaly detection
        if (model.online_detection(X.col(i))) {
            anomaly_count++;
            std::cout << anomaly_count << " anomaly found with delay (" << (i-tau) << ") and confidence (" << model.get_g() << ")" << std::endl;
            std::cout << "Past max confidence reached is: " << max_confidence << std::endl;
            if (model.get_g() > max_confidence) {
                max_confidence = model.get_g();
            }
            //return 0;
            model.reset_g();
        }
    }
    std::cout << "No anomaly found. Current confidence is: " << model.get_g() << std::endl;
    std::cout << "Max confidence reached is: " << max_confidence << std::endl;
    return 0;
} /* main */
