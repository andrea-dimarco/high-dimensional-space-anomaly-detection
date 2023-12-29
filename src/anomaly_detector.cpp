// Main

#include "include/gem.h"
#include "include/pca.h"
#include <iostream>
#include <assert.h>
#include <random>
#include <cstring>
#include <cstdlib>
#include <assert.h>
#include <fstream>

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

double GEM_objective_function(GEM model, double h, double alpha, Eigen::MatrixXd X) {

    double loss = 0;
    int N = X.cols(); // number of samples
    int anomaly_count = 0;
    
    for (int i = 0; i < N; i++) {
        // anomaly detection
        if (model.online_detection(X.col(i))) {
            //std::cout << anomaly_count << " anomalies found." << std::endl;
            anomaly_count++;
            model.reset_g();
        }
    }

    double FAR = double(anomaly_count) / double(N); // probability of 
    loss = FAR;

    return loss;
}

double PCA_objective_function(PCA model, double h, double alpha, Eigen::MatrixXd X) {

    double loss = 0;
    int N = X.cols(); // number of samples
    int anomaly_count = 0;
    
    for (int i = 0; i < N; i++) {
        // anomaly detection
        if (model.online_detection(X.col(i))) {
            //std::cout << anomaly_count << " anomalies found." << std::endl;
            anomaly_count++;
            model.reset_g();
        }
    }

    double FAR = double(anomaly_count) / double(N); // probability of 
    loss = FAR;

    return loss;
}

/**
 * Generate an Eigen matrix out of a previously populated file from the save_data() function
*/
Eigen::MatrixXd load_data(std::string file_path) {
	
	// the input is the file: "fileToOpen.csv":
	// a,b,c
	// d,e,f
	// This function converts input file data into the Eigen matrix format

	// the matrix entries are stored in this variable row-wise. For example if we have the matrix:
	// M=[a b c 
	//	  d e f]
	// the entries are stored as matrixEntries=[a,b,c,d,e,f], that is the variable "matrixEntries" is a row vector
	// later on, this vector is mapped into the Eigen matrix format
	std::vector<double> matrixEntries;
	
	std::ifstream matrixDataFile(file_path); // store the data from the matrix
	std::string matrixRowString; // store the row of the matrix that contains commas 
	std::string matrixEntry; // store the matrix entry

	// this variable is used to track the number of rows
	int matrixRowNumber = 0;

	while (std::getline(matrixDataFile, matrixRowString)) // here we read a row by row of matrixDataFile and store every line into the string variable matrixRowString
	{
		std::stringstream matrixRowStringStream(matrixRowString); //convert matrixRowString that is a string to a stream variable.

		while (std::getline(matrixRowStringStream, matrixEntry,',')) // here we read pieces of the stream matrixRowStringStream until every comma, and store the resulting character into the matrixEntry
		{
			matrixEntries.push_back(std::stod(matrixEntry));   //here we convert the string to double and fill in the row vector storing all the matrix entries
		}
		matrixRowNumber++; //update the column numbers
	}
	// here we conver the vector variable into the matrix and return the resulting object, 
	// note that matrixEntries.data() is the pointer to the first memory location at which the entries of the vector matrixEntries are stored;
	return Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> (matrixEntries.data(), matrixRowNumber, matrixEntries.size() / matrixRowNumber);
}

/**
 * Main function
 */
int main(int argc, char *argv[])
{
    bool use_GEM = false;
    double h = 5.0;
    double alpha = 0.2;
    bool is_offline = true;
    bool load = false; 
    std::string dataset_path;
    int p = 10;
    int N = 1000;

    // load arguments
    if(argc > 1) {
        if (*argv[1] == 'y') { use_GEM = true; }
        else { use_GEM = false; }

        if (argc > 2) { h = atof(argv[2]); }
        if (argc > 3) { alpha = atof(argv[3]); }

        if (argc > 4) {
            if (*argv[4] == 'y') { is_offline = true; }
            else { is_offline = false; }
        }
        if (argc > 5) {
            if (*argv[5] == 'y') { load = true; }
            else { load = false; }
        }
        if (load) { if (argc > 6) { dataset_path = std::string(argv[6]); } }
        else {
            if (argc > 6) { p = atoi(argv[6]); }
            if (argc > 7) { N = atoi(argv[7]); }
        }
    }

    // Load in dataset
    Eigen::MatrixXd X;
    if (load) {
        //std::cout << "Loading file: " << dataset_path << std::endl;
        X = load_data(dataset_path);
        p = X.rows(); N = X.cols();
    } else { // generate a random dataset
        X = random_dataset(p, N, false);
    }

    double loss;
    // Run model
    if (use_GEM) {
        GEM gem(p);
        if (is_offline) {
            gem.offline_phase(X);
            return 0;
        } else { // online phase
            gem.load_model();
            gem.set_h(h);
            gem.set_alpha(alpha);
            loss = GEM_objective_function(gem, h, alpha, X);
        }
    } else {
        PCA pca(p);
        if (is_offline) {
            pca.offline_phase(X);
            return 0;
        } else { // online phase
            pca.load_model();
            pca.set_h(h);
            pca.set_alpha(alpha);
            loss = PCA_objective_function(pca, h, alpha, X);
        }
    } 

    // // save results <- This creates issues when parallelizing
    // std::ofstream myfile;
    // myfile.open("loss.txt");
    // myfile << loss;
    // myfile.close();

    // return results
    std::cout << loss;

    return 0;
} /* main */
