#include "include/pca.h"

/**
 * Constructor
*/
PCA::PCA(int p, float partition_1/*=0.5*/, float partition_2/*=0.5*/,
        float gamma/*=0.95*/, double alpha/*=2.0*/, double h/*=5.0*/) {
    
    assert((gamma>0) && (alpha>0) && (h>0));
    assert(p>0);
    this->p = p;
    // Offline Phase 
    this->partition_1 = partition_1;
    this->partition_2 = partition_2;
    this->gamma = gamma;
    // Online Phase
    this->alpha = alpha;
    this->h = h;
    this->g = 0.0;
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
void PCA::compute_subsets(Eigen::MatrixXd X) {
    
    assert(X.rows() == this->p); // feature dimension must be on the y axis
    
    this->N  = X.cols();
    this->N1 = (int)(this->N*this->partition_1);
    this->N1 = (this->N1 <= 0) ? 1 : this->N1;
    this->N2 = (int)(this->N*this->partition_1);
    this->N2 = (this->N2 <= 0) ? 1 : this->N2;

    this->S1.resize(this->p,this->N1);
    this->S2.resize(this->p,this->N2);

    // Select random items
    Eigen::VectorXd indexes(N);
    for (int i = 0; i < N; i++) { indexes(i) = i; }

    // Get S1 
    std::random_shuffle(indexes.begin(), indexes.end());
    for (int i = 0; i < N1; i++) { S1.col(i) = X.col(indexes(i)); }

    // Get S2
    std::random_shuffle(indexes.begin(), indexes.end());
    for (int i = 0; i < N2; i++) { S2.col(i) = X.col(indexes(i)); }
} /* cumpute_subsets */

/**
 * Determine the principal subspace using Xstat_PCA
*/
void PCA::compute_pca() {
    this->baseline_distances.resize(this->N2);
    this->baseline_distances.setZero();

    this->baseline_mean_vector = S1.colwise().mean(); // mean for every feature (column wise)
}

/**
 * Compute baseline distances (residual magnitudes) using X_2 and the principal subspace
*/
void PCA::compute_baseline_distances() {

}

/**
 * Computes the covariance matrix given the datapoints from S1 and the sample mean
 * https://stackoverflow.com/questions/15138634/eigen-is-there-an-inbuilt-way-to-calculate-sample-covariance
*/
void PCA::compute_covariance_matrix() {
    Eigen::MatrixXd centered = centered = this->S1.rowwise() - this->baseline_mean_vector.transpose();
    this->covariance_matrix = (centered.adjoint() * centered) / double(this->N1);
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
 /* compute_summary_statistics */
/** 
 * Returns S1
*/
Eigen::MatrixXd PCA::get_S1() {
    return this->S1;
} /* get_S1 */
/** 
 * Returns S2
*/
Eigen::MatrixXd PCA::get_S2() {
    return this->S2;
} /* get_S2 */
/** 
 * Returns the baseline distances.
 * Don't forget to compute them in the OFFLINE phase!!
*/
Eigen::MatrixXd PCA::get_baseline_distances() {
    return this->baseline_distances;
} /* get_baseline_distances */

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

    PCA::compute_subsets(X);
    if (save_file) { PCA::save_baseline(file_path); }
} /* offline_phase */

bool PCA::online_detection(Eigen::VectorXd sample) {

    
    return false;
} /* online_detection */