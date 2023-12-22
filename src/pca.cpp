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
 * Given the set of nominal datapoints, divides data in two subsets S1 and S2 
 * with sizes N1 and N2 = |X| - N1 where N1 is a percentage of X
*/
Eigen::MatrixXd PCA::compute_subset(Eigen::MatrixXd X, int dim) {
    
    assert(X.rows() == this->p); // feature dimension must be on the y axis

    Eigen::MatrixXd S(this->p, dim);

    // Select random items
    Eigen::VectorXd indexes(this->N);
    for (int i = 0; i < this->N; i++) { indexes(i) = i; }
    std::random_shuffle(indexes.begin(), indexes.end());

    // Get S
    for (int i = 0; i < dim; i++) { S.col(i) = X.col(indexes(i)); }

    return S;
} /* cumpute_subsets */

/**
 * Determine the principal subspace using Xstat_PCA
*/
void PCA::compute_pca(Eigen::MatrixXd S1) {

    // compute the sample data mean vector
    this->baseline_mean_vector = S1.rowwise().mean(); // mean for every feature (column wise)
    std::cout << "Sample data mean computed" << std::endl;

    // get the sample data covariance matrix
    Eigen::MatrixXd covariance_matrix = PCA::compute_covariance_matrix(S1);
    std::cout << "Covariance matrix computed." << std::endl;
    // calculate eigen vectors and eigen vlaues

    // define matrix for eigenvalues and eigenvectors
    // can use type double instead of complex<double> because matrix is real symmetric (in theory)
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigen_value_solver(covariance_matrix);
    Eigen::VectorXd eigen_values_vector = eigen_value_solver.eigenvalues().real().col(0);
    Eigen::MatrixXd eigen_vector_matrix = eigen_value_solver.eigenvectors().real();
    std::cout << "Eigen components found." << std::endl;

    // based on gamma derive optimal r (subdimension value) 
    double sum_eigenvalues = eigen_values_vector.sum();
    int optimal_subdimension;
    double r_sum = 0;

    for (int i = p-1; i >= 0; i--) {
        // sum eigenvalues up until i
        r_sum += eigen_values_vector[i];
        // check if ratio is optimal i.e. ratio as close to equal to gamma
        if ((r_sum / sum_eigenvalues) >= gamma) {
            optimal_subdimension = p-i;
            break;
        }
    }
    std::cout << "Found best subdimension r: " << optimal_subdimension << std::endl;

    // get V = the eigen vectors corresponding to the r largest eigenvalues
    // we take every row but only the r (= optimal dimension) rightmost columns
    Eigen::MatrixXd V = eigen_vector_matrix(Eigen::all,Eigen::lastN(optimal_subdimension));
        
    // compute res_proj = Identity Matrix - (V * V')
    this->res_proj = Eigen::MatrixXd::Identity(p,p) - (V * V.adjoint());
}

/**
 * Computes baseline distances (residual magnitudes) using S2 and the principal subspace
*/
void PCA::compute_baseline_distances(Eigen::MatrixXd S2) {
    // for every datapoint in S2 compute the variance, then rowwise the L2 norm
    // should be a vector with the same cardinality as S2 
    // this is wrong I think
    assert(S2.cols() == this->N2);
    this->baseline_distances = ( this->res_proj *
                                    ( S2 - this->baseline_mean_vector.replicate(1,this->N2) ) )
                                            .colwise().squaredNorm();
    assert(this->baseline_distances.rows() == this->N2); 

    // sort the vector of l2 norms of residuals
    std::sort(this->baseline_distances.begin(),this->baseline_distances.end());
}

/**
 * Computes the covariance matrix given the datapoints from S1 and the sample mean
*/
Eigen::MatrixXd PCA::compute_covariance_matrix(Eigen::MatrixXd S1) {
    Eigen::MatrixXd centered = S1.colwise() - this->baseline_mean_vector;
    return (centered * centered.adjoint()) / double(this->N1);
} /* covariance matrix computation*/

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
                    std::string baseline_path/*="./baseline_distances.csv"*/,
                    std::string res_proj_path/*="./res_proj.csv"*/,
                    std::string mean_vector_path/*="./mean_vector.csv"*/,
                    std::string parameters_path/*="./parameters.csv"*/,
                    bool verbosity/*=false*/) {
    // sanity check
    assert(X.rows() == this->p);

    if (verbosity) { std::cout << "Starting offline phase" << std::endl; }
    
    assert(X.rows() == this->p);

    // compute dimensions
    this->N  = X.cols();
    this->N1 = (int)(this->N*this->partition_1);
    this->N1 = (this->N1 <= 0) ? 1 : this->N1;
    this->N2 = (int)(this->N*this->partition_2);
    this->N2 = (this->N2 <= 0) ? 1 : this->N2;

    Eigen::MatrixXd S1 = PCA::compute_subset(X, this->N1);
    Eigen::MatrixXd S2 = PCA::compute_subset(X, this->N2);
    if (verbosity) {
        std::cout << "Subsets computed" << std::endl;
        std::cout << "S1 cardinality " << S1.cols() << std::endl;
    }
    
    PCA::compute_pca(S1);
    if (verbosity) {
        std::cout << "Principal Components found\n";
        std::cout << "Proceeding to baseline statistics\n";
    }
    
    PCA::compute_baseline_distances(S2);
    if (save_file) { PCA::save_model(baseline_path, res_proj_path, mean_vector_path, parameters_path); }
} /* offline_phase */

/** Performs the ReLU function on the given input.
 * Which is max(0, x)
*/
double PCA::ReLU(double x) {
    return std::clamp(x, 0.0, x);
} /* ReLU */

/**
 * Counts the amount of element in the vector v that are greater than scalar
*/
int PCA::characteristic_function(Eigen::VectorXd v, double scalar) {
    int result = 0;
    for(int i = 0; i < v.size(); i++) {
        if (v(i) >= scalar) {
            result++;
        }
    }
    return result;
} /* characteristic_function */

bool PCA::online_detection(Eigen::VectorXd sample) {

    bool anomaly_found = false;
    
    // calculate residual term for the sample
    double residual_term = (this->res_proj * (sample - this->baseline_mean_vector)).squaredNorm();
    double tail_probability;

    // compute probability
    tail_probability = double(characteristic_function(this->baseline_distances, residual_term)) / double(this->N2);
    if (tail_probability == 0.0) { // special case check
        tail_probability = double(1 / this->N2);
    }

    // CUSUM
    this->g += log(this->alpha / tail_probability);
    this->g = PCA::ReLU(this->g);

    // anomaly check
    if (this->g >= this->h) {
        anomaly_found = true;
    } else {
        anomaly_found = false;
    }
    return anomaly_found;
} /* online_detection */

void PCA::save_res_proj(std::string file_path/*=./res_proj.csv"*/) {
    const Eigen::IOFormat CSVFormat(Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n");

	std::ofstream save_file(file_path);
	if (save_file.is_open()) {
		save_file << this->res_proj.format(CSVFormat);
		save_file.close();
	}
}
void PCA::load_res_proj(std::string file_path/*="./res_proj.csv"*/) {
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
	this->res_proj = Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> (matrixEntries.data(), matrixRowNumber, matrixEntries.size() / matrixRowNumber);
}

void PCA::save_mean_vector(std::string file_path/*="./mean_vector.csv"*/) {
    const Eigen::IOFormat CSVFormat(Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n");

	std::ofstream save_file(file_path);
	if (save_file.is_open()) {
		save_file << this->baseline_mean_vector.format(CSVFormat);
		save_file.close();
	}
}
void PCA::load_mean_vector(std::string file_path/*="./mean_vector.csv"*/) {
    
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
	this->baseline_mean_vector = Eigen::Map<Eigen::Vector<double, Eigen::Dynamic>> (values.data(), values.size());
}

void PCA::save_parameters(std::string file_path/*="./parameters.csv"*/) {
    
    const Eigen::IOFormat CSVFormat(Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n");

    Eigen::VectorXd parameters(6);

    parameters(0) = this->N1;
    parameters(1) = this->N2;
    parameters(2) = this->alpha;
    parameters(3) = this->p;
    parameters(4) = this->gamma;
    parameters(5) = this->h;
	std::ofstream save_file(file_path);
	if (save_file.is_open()) {
		save_file << parameters.format(CSVFormat);
		save_file.close();
	}
} /* save_parameters */
void PCA::load_parameters(std::string file_path/*="./parameters.csv"*/) {
    std::vector<double> values;
	
	std::ifstream load_file(file_path); // store the data from the matrix
	std::string row_string; // store the row of the matrix that contains commas 
	std::string value; // store the matrix entry

    Eigen::VectorXd parameters;

	while (std::getline(load_file, row_string)) // here we read a row by row of matrixDataFile and store every line into the string variable matrixRowString
	{
		std::stringstream row_stream(row_string); //convert matrixRowString that is a string to a stream variable.

		while (std::getline(row_stream, value, ',')) // here we read pieces of the stream row_stream until every comma, and store the resulting character into the matrixEntry
		{
			values.push_back(std::stod(value));   //here we convert the string to double and fill in the row vector storing all the matrix entries
		}
	}
	parameters = Eigen::Map<Eigen::Vector<double, Eigen::Dynamic>> (values.data(), values.size());
    this->N1 = parameters(0);
    this->N2 = parameters(1);
    this->alpha = parameters(2);
    this->p = parameters(3);
    this->gamma = parameters(4);
    this->h = parameters(5);
} /* load_parameters */


void PCA::save_model(std::string baseline_path/*="./baseline_distances.csv"*/,
                std::string res_proj_path/*="./res_proj.csv"*/,
                std::string mean_vector_path/*="./mean_vector.csv"*/,
                std::string parameters_path/*="./parameters.csv"*/) {
    PCA::save_baseline(baseline_path);
    PCA::save_res_proj(res_proj_path);
    PCA::save_mean_vector(mean_vector_path);
    PCA::save_parameters(parameters_path);
}
void PCA::load_model(std::string baseline_path/*="./baseline_distances.csv"*/,
                std::string res_proj_path/*="./res_proj.csv"*/,
                std::string mean_vector_path/*="./mean_vector.csv"*/,
                std::string parameters_path/*="./parameters.csv"*/) {
    PCA::load_baseline(baseline_path);
    PCA::load_res_proj(res_proj_path);
    PCA::load_mean_vector(mean_vector_path);
    PCA::load_parameters(parameters_path);
}

void PCA::reset_g() {
    this->g = 0.0;
}

double PCA::get_g() {
    return this->g;
}