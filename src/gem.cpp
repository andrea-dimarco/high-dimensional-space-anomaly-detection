#include "include/gem.h"

/**
 * Constructor
*/
GEM::GEM(int p, int k/*=4*/,double alpha/*=0.2*/, double h/*=5.0*/) {
    
    assert((k>0) && (alpha>0) && (h>0));
    assert(p>0);
    this->k = k;
    this->alpha = alpha;
    this->h = h;
    this->p = p;
    this->g = 0.0;
    
    // Online Phase
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
void GEM::partition_data(Eigen::MatrixXd X, float partition/*=0.15*/) {
    
    assert(X.rows() == this->p); // feature dimension must be on the y axis
    assert((partition>0.0) && (partition<1.0));
    
    this->N  = X.cols();
    this->N1 = (int)(this->N*partition);
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
Eigen::MatrixXd GEM::get_S1() {
    return this->S1;
}
/** 
 * Returns S2
*/
Eigen::MatrixXd GEM::get_S2() {
    return this->S2;
}
/** 
 * Returns the baseline distances.
 * Don't forget to compute them in the OFFLINE phase!!
*/
Eigen::MatrixXd GEM::get_baseline_distances() {
    return this->baseline_distances;
}

/**
 * Resets the g value to 0.0, representing the amount (and intensity) of the outliers found.
*/
void GEM::reset_g() {
    this->g = 0.0;
}
/**
 * Changes the stored h value representing the threshold for the anomaly detection.
*/
void GEM::set_h(double h) {
    this->h = h;
}
/**
 * Changes the parameter alpha.
*/
void GEM::set_alpha(double alpha) {
    this->alpha = alpha;
}

/**
 * Computes the k Nearest Neighbors of the set S2 in the set S1.
*/
void GEM::kNN() {
    // if these fail, you need to call GEM::partition_data(X) before kNN()!!
    assert((this->N == (this->N1 + this->N2)));
    assert(this->baseline_distances.size() == this->N2);
    assert(this->S1.cols() == this->N1);
    assert(this->S2.cols() == this->N2);

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

/**
 * Save the computed baseline in a .csv file
*/
void GEM::save_baseline(std::string file_path/*="./baseline_distances.csv"*/) {
    
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
void GEM::load_baseline(std::string file_path/*="./baseline_distances.csv"*/) {
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

void GEM::save_parameters(std::string file_path/*="./parameters.csv"*/) {
    
    const Eigen::IOFormat CSVFormat(Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n");

    Eigen::VectorXd parameters(6);

    parameters(0) = this->N1;
    parameters(1) = this->N2;
    parameters(2) = this->alpha;
    parameters(3) = this->p;
    parameters(4) = this->k;
    parameters(5) = this->h;
    
	std::ofstream save_file(file_path);
	if (save_file.is_open()) {
		save_file << parameters.format(CSVFormat);
		save_file.close();
	}
} /* save_parameters */
void GEM::load_parameters(std::string file_path/*="./parameters.csv"*/) {
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
    this->k = parameters(4);
    this->h = parameters(5);
} /* load_parameters */

void GEM::save_S1(std::string file_path/*="./S1.csv"*/) {
    const Eigen::IOFormat CSVFormat(Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n");

	std::ofstream save_file(file_path);
	if (save_file.is_open()) {
		save_file << this->S1.format(CSVFormat);
		save_file.close();
	}
} /* save_S1 */
void GEM::load_S1(std::string file_path/*="./S1.csv"*/) {
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
	this->S1 = Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> (matrixEntries.data(), matrixRowNumber, matrixEntries.size() / matrixRowNumber);
}
/**
 * Save the model offline computed baseline and parameters
*/
void GEM::save_model(std::string baseline_path/*="./baseline_distances.csv"*/,
                std::string parameters_path/*="./parameters.csv"*/,
                std::string S1_path/*="./S1.csv"*/) {
    save_baseline(baseline_path);
    save_parameters(parameters_path);
    save_S1(S1_path);
}
/**
 * Load the model offline computed baseline and parameters
*/
void GEM::load_model(std::string baseline_path/*="./baseline_distances.csv"*/,
                std::string parameters_path/*="./parameters.csv"*/,
                std::string S1_path/*="./S1.csv"*/) {
    load_baseline(baseline_path);
    load_parameters(parameters_path);
    load_S1(S1_path);
}


/**
 * The whole offline phase of the GEM model.
*/
void GEM::offline_phase(Eigen::MatrixXd X, float partition/*=0.15*/,
                    bool save_file/*=true*/, std::string baseline_path/*="./baseline_distances.csv"*/,
                    std::string parameters_path/*="./parameters.csv"*/) {
    // sanity check
    assert(X.rows() == this->p);

    GEM::partition_data(X, partition); // returns S1, S2
    GEM::kNN();// takes S1, S2 in input
    if (save_file) { GEM::save_model(baseline_path, parameters_path); }
} /* offline_phase */

/**
 * Counts the amount of element in the vector v that are greater than scalar
*/
int GEM::characteristic_function(Eigen::VectorXd v, double scalar) {
    int result = 0;
    for(int i = 0; i < v.size(); i++) {
        if (v(i) >= scalar) {
            result++;
        }
    }
    return result;
} /* characteristic_function */

/**
 * One-step online analysis for anomalies.
*/
bool GEM::online_detection(Eigen::VectorXd sample) {

    bool anomaly_found = false;
    Eigen::VectorXd S1_sample;
    Eigen::VectorXd tmp_distances(this->N1);
    double dist, k_sum, tail_probability;
    int i;

    for (i = 0; i < this->N1; i++) {
        S1_sample = this->S1.col(i);
        dist = GEM::euclidean_dist(sample, S1_sample);
        tmp_distances(i) = dist;
    }
     // sum the best k neighbors
    std::sort(tmp_distances.begin(), tmp_distances.end());
    k_sum = 0;
    for (i = 0; (i < this->k) && (i < this->N1); i++) { k_sum += tmp_distances(i); }

    // compute probability
    tail_probability = ((double)characteristic_function(this->baseline_distances, k_sum)) / N2;
    if (tail_probability == 0.0) {
        tail_probability = (double) 1 / N2;
    }

    // CUSUM
    this->g += log(this->alpha / tail_probability);

    // anomaly check
    if (this->g >= this->h) {
        anomaly_found = true;
    } else {
        anomaly_found = false;
    }
    return anomaly_found;
} /* online_detection */