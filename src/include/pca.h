// Geometric Entropy Minimization

#pragma once

#include <iostream>
#include <random>
#include <cstring>
#include <cstdlib>
#include <assert.h>
#include <cmath>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/src/Core/Map.h>
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
    Eigen::MatrixXd S1;
    Eigen::MatrixXd S2;
    Eigen::VectorXd baseline_distances;

    Eigen::VectorXd baseline_mean_vector;
    Eigen::MatrixXd covariance_matrix;

    int N, N1, N2;   // set partition dimensions
    float partition_1, partition_2; // dimensions of S1, S2
    float gamma;     // precision to be achieved by PCA
    double alpha;    // outlier detection hyperparameter
    double g;        // amount of outliers discovered so far
    int p;           // dimensionality of the space
    int k;           // neighbors to consider
    double h;        // anomaly detection threshold
    
public:
    PCA(int p, float partition_1=0.5, float partition_2=0.5, float gamma=0.95, double alpha=0.2, double h=5.0);

    double l2_norm(Eigen::VectorXd p1);

    void compute_subsets(Eigen::MatrixXd X);
    void compute_pca();
    void compute_baseline_distances();

    Eigen::MatrixXd get_S1();
    Eigen::MatrixXd get_S2();
    Eigen::MatrixXd get_baseline_distances();

    void save_baseline(std::string file_path="./baseline_distances.csv");
    void load_baseline(std::string file_path="./baseline_distances.csv");

    void compute_covariance_matrix();
    void compute_summary_statistics();

    void offline_phase(Eigen::MatrixXd X, bool strict_k=false, bool save_file=true, std::string file_path="./baseline_distances.csv");

    // return true if anomaly found
    bool online_detection(Eigen::VectorXd sample);
};