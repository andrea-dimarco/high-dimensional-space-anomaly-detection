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

class GEM {

private:
    Eigen::MatrixXd S1;
    Eigen::MatrixXd S2;
    Eigen::VectorXd baseline_distances;
    int N, N1, N2;   // set partition dimensions
    double g;        // current level of outliers detected 
    double alpha;    // outlier detection hyperparameter
    int p;           // dimensionality of the space
    int k;           // neighbors to consider
    double h;        // anomaly detection threshold
    
public:
    GEM(int p, int k=4, double alpha=0.2, double h=5.0);

    double euclidean_dist(Eigen::VectorXd p1, Eigen::VectorXd p2);

    double ReLU(double x);

    void partition_data(Eigen::MatrixXd X, float partition=0.15);

    Eigen::MatrixXd get_S1();
    Eigen::MatrixXd get_S2();
    Eigen::MatrixXd get_baseline_distances();

    void reset_g();
    void set_h(double h);
    void set_alpha(double alpha);

    Eigen::MatrixXd random_permutation(Eigen::MatrixXd X, bool columns=true, bool index_check=true);

    void kNN();

    void save_model(std::string baseline_path="./baseline_distances.csv",
                    std::string parameters_path="./parameters.csv");
    void load_model(std::string baseline_path="./baseline_distances.csv",
                    std::string parameters_path="./parameters.csv");

    void save_baseline(std::string file_path="./baseline_distances.csv");
    void load_baseline(std::string file_path="./baseline_distances.csv");

    void save_parameters(std::string file_path="./parameters.csv");
    void load_parameters(std::string file_path="./parameters.csv");

    int characteristic_function(Eigen::VectorXd v, double scalar);

    void offline_phase(Eigen::MatrixXd X, float partition=0.15,
                        bool save_file=true, std::string baseline_path="./baseline_distances.csv",
                        std::string parameters_path="./parameters.csv");

    // return true if anomaly found
    bool online_detection(Eigen::VectorXd sample);
};