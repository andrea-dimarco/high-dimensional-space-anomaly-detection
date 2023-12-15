#include "include/gem.h"

/**
 * Constructor
*/
GEM::GEM(int p, int k/*=4*/,double alpha/*=2.0*/, double h/*=5.0*/,float partition/*=0.15*/) {
    assert((partition>0.0) && (partition<1.0));
    assert((k>0) && (alpha>0) && (h>0));
    assert(p>0);
    this->k = k;
    this->alpha = alpha;
    this->h = h;
    this->partition = partition;
    // Offline Phase 
    // Partition dataset
    // calculate sum kNN for each yj in S1 for each xj in S2
    // sort dj vector in ascending order
    this->p = p;
} /* GEM */

/** 
 * Given two vectors, returns the euclidean distance between them.
*/
double GEM::euclidean_dist(Eigen::VectorXd p1, Eigen::VectorXd p2) {
    assert(p1.size() == p2.size());
    return sqrt((p1-p2).dot((p1-p2)));
} /* euclidean_dist */

/** Performs the ReLU function on the given input.
 * Which is max(0, x)
*/
double GEM::ReLU(double x) {
    return std::clamp(x, 0.0, x);
} /* ReLU */

/**
 * Given the set of nominal datapoints, randomly partitions the data in the two sets S1 and S2.
 * TODO: find better way of randomly selecting samples, like generating M uniformly random indexes to define one of the two partitions?
*/
void GEM::partition_data(Eigen::MatrixXd X) {
    
    assert(X.rows() == this->p); // feature dimension must be on the y axis

    this->N  = X.cols();
    this->N1 = (int)(this->N*this->partition);
    this->N1 = (this->N1 <= 0) ? 1 : this->N1;
    this->N2 = this->N - this->N1;

    this->S1.resize(this->p,this->N1);
    this->S2.resize(this->p,this->N2);
    this->baseline_distances.resize(this->N2);
    this->baseline_distances.setZero();

    // Don't forget the random shuffle!!
    Eigen::MatrixXd X_perm = GEM::random_permutation(X);

    int i, j, l;
    i = 0;
    for (j = 0; i < N1; j++) {
        S1.col(j) = X_perm.col(i);
        i++;
    }
    for (l = 0; l < N2; l++) {
        S2.col(l) = X_perm.col(i);
        i++;
    }
    assert(i == N); // we got every element
} /* partition_data */

/**
 * Randomly permutates either the columns or the rows of the given matrix.
 * Depending on the assignment of the boolean variable columns:
 *  columns = true  -> permutates columns
 *  columns = false -> permutate rows
 * Also:
 *  index_check = true  -> will NEVER try to swap the column with itself
 *  index_check = false -> might swap the column with itseld (so no swap)
*/
Eigen::MatrixXd GEM::random_permutation(Eigen::MatrixXd X, bool columns/*=true*/, bool index_check/*=true*/) {
    // Sanity check
    if (columns) {
        if (X.cols() == 1) {
            std::cout << "Are you kidding me??" << std::endl;
            return X;
        }
    } else {
        if (X.rows() == 1) {
            std::cout << "Are you kidding me??" << std::endl;
            return X;
        }
    }

    // Define a PRG
    std::default_random_engine engine(0);
    std::bernoulli_distribution bernoulli;
    Eigen::MatrixXd X_perm = X;
    int i, swap_index;
    Eigen::MatrixXd tmp;

    // Permutate
    if (columns) {
        std::uniform_int_distribution<> uniform_cols(0, X.cols()-1);
        // randomly permutate columns
        for (i = 0; i < X.cols(); i++) {
            if (bernoulli(engine)) {
                // swap column with another random column
                do {
                    swap_index = uniform_cols(engine);
                } while((swap_index == i) && index_check); // don't swap the vector with itself!!
                tmp = X_perm.col(i);
                X_perm.col(i) = X_perm.col(swap_index);
                X_perm.col(swap_index) = tmp;
            }
        }
    } else {
        std::uniform_int_distribution<> uniform_rows(0, X.rows()-1);
        // randomly permutate rows
        for (i = 0; i < X.rows(); i++) {
            if (bernoulli(engine)) {
                // swap column with another random column
                do {
                    swap_index = uniform_rows(engine);
                } while((swap_index == i) && index_check); // don't swap the vector with itself!!
                tmp = X_perm.row(i);
                X_perm.row(i) = X_perm.row(swap_index);
                X_perm.row(swap_index) = tmp;
            }
        }
    }
    return X_perm;
} /* random_permutation */

/** 
 * Returns S1
*/
Eigen::MatrixXd GEM::getS1() {
    return this->S1;
}
/** 
 * Returns S2
*/
Eigen::MatrixXd GEM::getS2() {
    return this->S2;
}
/** 
 * Returns the baseline distances.
 * Don't forget to compute them in the OFFLINE phase!!
*/
Eigen::MatrixXd GEM::getBaselineDistances() {
    return this->baseline_distances;
}

/**
 * Computes the k Nearest Neighbors of the set S2 in the set S1.
 * if strict_k = true  it will return an error if the set S1 doesn't have enough neighbors
 * else                it will compute the baseline with less than k neighbors if S1 is small
*/
void GEM::kNN(bool strict_k/*=false*/) {
    // if these fail, you need to call GEM::partition_data(X) before kNN()!!
    assert((this->N == (this->N1 + this->N2)));
    assert(this->baseline_distances.size() == this->N2);
    assert(this->S1.cols() == this->N1);
    assert(this->S2.cols() == this->N2);
    // not enough elements in the set to compute the k neighbors
    assert((this->k <= this->N1) || !(strict_k));

    Eigen::MatrixXd S1_sample, S2_sample;
    Eigen::VectorXd tmp_distances(this->N1);
    double dist, k_sum;
    int i, j;
    for (i = 0; i < this->N2; i++) {
        // compute distances
        S2_sample = this->S2.col(i);
        tmp_distances.setZero();
        for (j = 0; j < this->N1; j++) {
            S1_sample = this->S1.col(j);
            dist = GEM::euclidean_dist(S1_sample,S2_sample);
            tmp_distances(j) = dist;
        }
        // sum the best k neighbors
        std::sort(tmp_distances.begin(), tmp_distances.end());
        k_sum = 0;
        for (j = 0; (j < this->k) && (j < this->N1); j++) { k_sum += tmp_distances(j); }
        // store value
        this->baseline_distances(i) = k_sum;
    }
} /* kNN */

