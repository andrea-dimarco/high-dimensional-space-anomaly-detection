// Geometric Entropy Minimization

#pragma once

#include <iostream>
#include <random>
#include <cstring>
#include <cstdlib>
#include <assert.h>
#include <cmath>
#include <Eigen/Dense>
#include <Eigen/src/Core/Map.h>
#include <fstream>

class PCA {
/**
 * Divide Data in subsets S1 and S2 with sizes N1 and N2
 * Compute sample mean x_mean and covariance matrix Q on S1
 * Compute eigenvalues and eigenvectors on Q
 * Compute gamma 
 * Compute r based on gamma and form matrix V
*/
private:
    Eigen::VectorXd baseline_distances, baseline_mean_vector;
    Eigen::MatrixXd res_proj;

    int N, N1, N2;   // set partition dimensions
    float partition_1, partition_2; // dimensions of S1, S2
    float gamma;     // precision to be achieved by PCA
    double alpha;    // outlier detection hyperparameter
    double g;        // amount of outliers discovered so far
    int p;           // dimensionality of the space
    double h;        // anomaly detection threshold
    
public:
    PCA(int p, float partition_1=0.5, float partition_2=0.5, float gamma=0.95, double alpha=0.2, double h=7.5);

    Eigen::MatrixXd compute_subset(Eigen::MatrixXd X, int dim);
    void compute_pca(Eigen::MatrixXd S1);
    void compute_baseline_distances(Eigen::MatrixXd S2);
    int characteristic_function(Eigen::VectorXd v, double scalar);

    Eigen::MatrixXd get_baseline_distances();

    double ReLU(double x);

    double get_g();
    void reset_g();

    void save_baseline(std::string file_path="./baseline_distances.csv");
    void load_baseline(std::string file_path="./baseline_distances.csv");

    void save_res_proj(std::string file_path="./res_proj.csv");
    void load_res_proj(std::string file_path="./res_proj.csv");

    void save_mean_vector(std::string file_path="./mean_vector.csv");
    void load_mean_vector(std::string file_path="./mean_vector.csv");

    void save_parameters(std::string file_path="./parameters.csv");
    void load_parameters(std::string file_path="./parameters.csv");

    void save_model(std::string baseline_path="./baseline_distances.csv",
                    std::string res_proj_path="./res_proj.csv",
                    std::string mean_vector_path="./mean_vector.csv",
                    std::string parameters_path="./parameters.csv");
    void load_model(std::string baseline_path="./baseline_distances.csv",
                    std::string res_proj_path="./res_proj.csv",
                    std::string mean_vector_path="./mean_vector.csv",
                    std::string parameters_path="./parameters.csv");

    Eigen::MatrixXd compute_covariance_matrix(Eigen::MatrixXd S1);

    void offline_phase(Eigen::MatrixXd X, bool strict_k=false, bool save_file=true,
                        std::string baseline_path="./baseline_distances.csv",
                        std::string res_proj_path="./res_proj.csv",
                        std::string mean_vector_path="./mean_vector.csv",
                        std::string parameters_path="./parameters.csv",
                        bool verbosity=false);

    // return true if anomaly found
    bool online_detection(Eigen::VectorXd sample);
};