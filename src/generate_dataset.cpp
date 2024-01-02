// Main

#include <iostream>
#include <random>
#include <cstring>
#include <cstdlib>
#include <fstream>
#include <Eigen/Dense>


void save_matrix(Eigen::MatrixXd X, std::string file_path) {
    const Eigen::IOFormat CSVFormat(Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n");

	std::ofstream save_file(file_path);
	if (save_file.is_open()) {
		save_file << X.format(CSVFormat);
		save_file.close();
	}
} /* save_matrix */

/**
 * Generate a matrix of random values given 5 parameters
 * Dimension on x axis,
 * Dimension on y axis,
 * is_uniform use a uniform distribution (otherwisen it will be Gaussian)
 * lower_bound (or mean)
 * upper_bound (or standard deviation)
 * seed (the seed for the rasndom generator)
*/
Eigen::MatrixXd random_dataset(int dim0=2, int dim1=2,
                                bool is_uniform=true, double a=0.0, double b=1.0,
                                int seed=0) {
    
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
 * Main function
 * is_uniform : chooses if the distribution is uniform or Gaussian
 * param1 : first parameter of the distribution (mean if gaussian, left bound if uniform)
 * param2 : second parameter of the distribution (variance if gaussian, right bound if uniform)
 * p : dimensionality of a sample 
 * N : number of samples to generate
 * dataset_path : the file in which to save the dataset
 */
int main(int argc, char *argv[])
{
    bool is_uniform = false;
    double param1 = 0.0;
    double param2 = 1.0; 
    int p = 10;
    int N = 1000;
    std::string dataset_path = "./dataset.csv";

    // load arguments
    if(argc > 1) {
        if (*argv[1] == 'y') { is_uniform = true; }
        else { is_uniform = false; }

        if (argc > 2) { param1 = atof(argv[2]); }
        if (argc > 3) { param2 = atof(argv[3]); }
        if (argc > 4) { p = atoi(argv[4]); }
        if (argc > 5) { N = atoi(argv[5]); }
        if (argc > 6) { dataset_path = std::string(argv[6]); }
    }

    // Create dataset
    Eigen::MatrixXd X = random_dataset(p, N, is_uniform, param1, param2);
    // Save dataset
    save_matrix(X, dataset_path);

    return 0;
} /* main */
