#include "include/pca.h"

/**
 * Constructor
*/
PCA::PCA(int p, int k/*=4*/,double alpha/*=2.0*/, double h/*=5.0*/,float partition) {
    
    assert((k>0) && (alpha>0) && (h>0));
    assert(p>0);
    this->k = k;
    this->partition = partition;
    this->alpha = alpha;
    this->h = h;
    this->p = p;
    // Offline Phase 
    // Online Phase
} /* PCA */

/** 
 * Given two vectors, returns the euclidean distance between them.
*/
double PCA::l2_norm(Eigen::VectorXd p1) {
    assert(p1.size() > 0);
    return p1.squaredNorm();
} /* euclidean_dist */

/**
 * Given the set of nominal datapoints, divides data in two subsets S1 and S2 
 * with sizes N1 and N2 = |X| - N1 where N1 is a percentage of X
*/
void PCA::divide_nominal_dataset(Eigen::MatrixXd X, float partition) {
    
    assert(X.rows() == this->p); // feature dimension must be on the y axis
    
    this->N  = X.cols();
    this->N1 = (int)(this->N*this->partition);
    this->N1 = (this->N1 <= 0) ? 1 : this->N1;
    this->N2 = this->N - this->N1;

    this->S1.resize(this->p,this->N1);
    this->S2.resize(this->p,this->N2);
    this->baseline_distances.resize(this->N2);
    this->baseline_distances.setZero();

    this->baseline_mean_vector = S1.colwise().mean(); // Is this colwise or rowwise? Must check!

    Eigen::MatrixXd X_perm = X;
    // TODO: generate N1 random indexes from |X| elements

    // set values of S1 and S2
    // For now we just split the data and S1 gets the first N1 elements of N
    for (int i = 0; i < N; i++) {
        if (i < N1) {
            S1.col(i) = X_perm.col(i);
        }
        else {
            S2.col(i - N1) = X_perm.col(i);
        }
        i++;
    }

} /* partition_data */

/**
 * Computes the covariance matrix given the datapoints from S1 and the sample mean
 * Should we trust stackoverflow?
 * https://stackoverflow.com/questions/15138634/eigen-is-there-an-inbuilt-way-to-calculate-sample-covariance
*/
void PCA::compute_covariance_matrix() {
    Eigen::MatrixXd centered = S2.rowwise() - baseline_mean_vector;
    this->covariance_matrix = (centered.adjoint() * centered) / double(S2.rows() - 1);
} /* covariance matrix computation*/

/**
 * Computes summary statistics of the PCA based algorithm 
 * i.e. the set of the l2 of the residual terms for each datapoint in S2
 * 
*/
void PCA::compute_summary_statistics() {
    // for each datapoint in s2 
        // compute rj 
        //compute the norm of rj
    // sort r_set in ascending order
    // assign to object vector
}

/** 
 * Returns S1
*/
Eigen::MatrixXd PCA::getS1() {
    return this->S1;
}
/** 
 * Returns S2
*/
Eigen::MatrixXd PCA::getS2() {
    return this->S2;
}
/** 
 * Returns the baseline distances.
 * Don't forget to compute them in the OFFLINE phase!!
*/
Eigen::MatrixXd PCA::getBaselineDistances() {
    return this->baseline_distances;
}

/**
 * Save the computed baseline in a .csv file
*/
void PCA::save_baseline(std::string file_path/*="./baseline_distances.csv"*/) {
    
    const Eigen::IOFormat CSVFormat(Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n");

	std::ofstream save_file(file_path);
	if (save_file.is_open()) {
		save_file << this->baseline_distances.format(CSVFormat);
		save_file.close();
	}
} /* save_baseline */
/**
 * Load the previously computed baselines from a csv file
*/
void PCA::load_baseline(std::string file_path/*="./baseline_distances.csv"*/) {
    std::vector<double> values;
	
	std::ifstream load_file(file_path); // store the data from the matrix
	std::string row_string; // store the row of the matrix that contains commas 
	std::string value; // store the matrix entry

	while (std::getline(load_file, row_string)) // here we read a row by row of matrixDataFile and store every line into the string variable matrixRowString
	{
		std::stringstream row_stream(row_string); //convert matrixRowString that is a string to a stream variable.

		while (std::getline(row_stream, value, ',')) // here we read pieces of the stream row_stream until every comma, and store the resulting character into the matrixEntry
		{
			values.push_back(std::stod(value));   //here we convert the string to double and fill in the row vector storing all the matrix entries
		}
	}
	// here we conver the vector variable into the matrix and return the resulting object, 
	// note that values.data() is the pointer to the first memory location at which the entries of the vector matrixEntries are stored;
	this->baseline_distances = Eigen::Map<Eigen::Vector<double, Eigen::Dynamic>> (values.data(), values.size());
} /* load_baseline */

/**
 * The whole offline phase of the PCA model.
*/
void PCA::offline_phase(Eigen::MatrixXd X,
                    bool strict_k/*=false*/, bool save_file/*=true*/, 
                    std::string file_path/*="./baseline_distances.csv"*/) {
    // sanity check
    assert(X.rows() == this->p);

    PCA::divide_nominal_dataset(X, partition);
    if (save_file) { PCA::save_baseline(file_path); }
}

bool PCA::online_detection(Eigen::VectorXd sample) {

    bool anomaly_found = false;


    return anomaly_found;
}