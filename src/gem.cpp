#include "include/gem.h"

GEM::GEM(int p, int k/*=4*/,double alpha/*=2.0*/, double h/*=5.0*/,float partition/*=0.15*/) {
    assert((partition>0.0) && (partition<1.0));
    assert((k>0) && (alpha>0) && (h>0));
    assert(p>0);
    this->k = k;
    this->alpha = alpha;
    this->h = h;
    this->partition = partition;

    this->p = p;
} /* GEM */

/** Given two vectors, returns the euclidean distance between them.
 * 
*/
double GEM::euclidean_dist(Eigen::VectorXd p1, Eigen::VectorXd p2) {
    assert(p1.size() == p2.size());
    return sqrt((p1-p2).dot((p1-p2)));
} /* euclidean_dist */

/** Performs the ReLU function on the given input.
 * Which is max(0, x)
*/
double GEM::ReLU(double x) {
    if (x < 0.0) { return 0.0; }
    else { return x; }
} /* ReLU */

/**
 * Given the set of nominal datapoints, partitions the data in the two sets S1 and S2.
 * TODO: implement the random shuffle
*/
void GEM::partition_data(Eigen::MatrixXd X) {
    
    assert(X.cols() == this->p); // feature dimension must be on the y axis

    this->N = X.rows();
    this->N1 = (int)(this->N*this->partition);
    this->N2 = this->N - this->N1;

    S1.resize(this->N1,this->p);
    S2.resize(this->N2,this->p);

    // Don't forget the random shuffle!!
    // do we really have to permute to produce the 2 partitions?
    // what about generating M uniformly random indexes to define one of the two partitions?
    Eigen::MatrixXd X_perm = GEM::random_permutation(X);

    int i, j, l;
    i = 0;
    for (j = 0; i < N1; j++) {
        S1.row(j) = X_perm.row(i);
        i++;
    }
    for (l = 0; l < N2; l++) {
        S2.row(l) = X_perm.row(i);
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
