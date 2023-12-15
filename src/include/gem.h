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

class GEM {

private:
    Eigen::MatrixXd S1;
    Eigen::MatrixXd S2;
    Eigen::VectorXd baseline_distances;
    int N, N1, N2;   // set partition dimensions
    float partition; // N1 should be 15% of N
    double alpha;    // outlier detection hyperparameter
    int p;           // dimensionality of the space
    int k;           // neighbors to consider
    double h;        // anomaly detection threshold
    
public:
    GEM(int p, int k=4,double alpha=2.0, double h=5.0,float partition=0.15);

    double euclidean_dist(Eigen::VectorXd p1, Eigen::VectorXd p2);

    double ReLU(double x);

    void partition_data(Eigen::MatrixXd X);

    Eigen::MatrixXd getS1();
    Eigen::MatrixXd getS2();
    Eigen::MatrixXd getBaselineDistances();

    Eigen::MatrixXd random_permutation(Eigen::MatrixXd X, bool columns=true, bool index_check=true);

    void kNN(bool strict_k=false);

    Eigen::VectorXd get_k_best(Eigen::VectorXd neighbors);

    double CUSUM();

    double tail_probability();

    void offline_phase();

    // return true if anomaly found
    bool online_detection();
};