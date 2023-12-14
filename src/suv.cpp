#include "include/suv.h"


SUV::SUV() {
    this->n = 0;
    this->m = 0;
    this->r = 0;

    A.resize(1,1); A.setZero();
    B.resize(1,1); B.setZero();
    C.resize(1,1); C.setZero();

    x0.resize(1,1); x0.setZero();

    old_x.resize(1,1); old_x.setZero();
    x.resize(1,1); x.setZero();
    y.resize(1,1); y.setZero();
    current_time = 0;

    input_sequence.resize(1,1); input_sequence.setZero();
    output_sequence.resize(1,1); output_sequence.setZero();
    state_sequence.resize(1,1); state_sequence.setZero();
    time_row.resize(1,1); time_row.setZero();

    this->simulation_ended = false;
}
SUV::SUV(Eigen::MatrixXd Am, Eigen::MatrixXd Bm, Eigen::MatrixXd Cm,
        Eigen::MatrixXd initial_state, Eigen::MatrixXd input_sequence_m) {
        
    this->A = Am; this-> B = Bm; this->C = Cm; this->x0 = initial_state; this->input_sequence = input_sequence_m;
    this->n = A.rows();
    this->m = B.cols();
    this->r = C.rows();
    this->N = this->input_sequence.cols();
    this->output_sequence.resize(r,N); this->output_sequence.setZero();
    this->state_sequence.resize(n,N);  this->state_sequence.setZero();
    this->time_row.resize(1,N);
    for (int i = 0; i < N; i++) {
        this->time_row(0,i) = (i+1); // count from 1
    }
    this->x = initial_state;
    this->old_x = initial_state;
    this->y.resize(r,1); y.setZero();
    this->current_time = 0;

    this->simulation_ended = false;
}

/**
 * Simulate N steps of the SUV.
*/
void SUV::run_simulation(bool store_history/*=true*/, bool verbose/*=false*/) {
    if (store_history) {
        for (int j = 0; j < this->N; j++) {
            if (j == 0) {
                this->state_sequence.col(j) = this->x0;
                this->output_sequence.col(j) = this->C*this->x0;
            } else {
                this->state_sequence.col(j) = this->A*this->state_sequence.col(j - 1) + this->B*this->input_sequence.col(j - 1);
                this->output_sequence.col(j) = this->C*this->state_sequence.col(j);
            }
            //current_time = j;	
        }
    } else {
        for (int j = 0; j < this->N; j++) {
            if (j == 0) {
                this->x = this->x0;
                this->old_x = this->x0;
                this->y = this->C*this->x0;
            } else {
                this->x = this->A*this->old_x + this->B*this->input_sequence.col(j - 1);
                this->y = this->C*this->x;

                this->old_x = this->x;
            }
            //current_time = j;
        }
    }
    this->simulation_ended = true;
    this->current_time = N-1;
}

/**
 * Simulate t steps of the SUV.
*/
void SUV::run_simulation_up_to(int t, bool store_history/*=true*/, bool verbose/*=false*/) {
    if (store_history) {
        for (int j = 0; j < t; j++) {
            if (j == 0) {
                this->state_sequence.col(j) = this->x0;
                this->output_sequence.col(j) = this->C*this->x0;
            } else {
                this->state_sequence.col(j) = this->A*this->state_sequence.col(j - 1) + this->B*this->input_sequence.col(j - 1);
                this->output_sequence.col(j) = this->C*this->state_sequence.col(j);
            }	
        }
    } else {
        for (int j = 0; j < t; j++) {
            if (j == 0) {
                this->x = this->x0;
                this->old_x = this->x0;
                this->y = this->C*this->x0;
            } else {
                this->x = this->A*this->old_x + this->B*this->input_sequence.col(j - 1);
                this->y = this->C*this->x;

                this->old_x = this->x;
            }	
        }
    }
    this->current_time = t;
}

/**
 * Returns the output vector at time t.
 * If you have already simulated with no history you will have to simulate again to get the output
*/
Eigen::MatrixXd SUV::get_output_at_time(int t, bool store_history/*=true*/, bool verbose/*=false*/) {
    if (!simulation_ended) {
        SUV::run_simulation_up_to(t, store_history, verbose);
    }
    return this->output_sequence.col(t-1);
}

/**
 * Save the most valuable variables of the SUV into separate files
*/
void SUV::save_data(std::string A_file_path, std::string B_file_path,
                        std::string C_file_path, std::string x0_file_path,
                        std::string input_sequence_file_path, std::string output_sequence_file_path) {
    
    const Eigen::IOFormat CSVFormat(Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n");

	std::ofstream A_file(A_file_path);
	if (A_file.is_open()) {
		A_file << this->A.format(CSVFormat);
		A_file.close();
	}
    std::ofstream B_file(B_file_path);
	if (B_file.is_open()) {
		B_file << this->B.format(CSVFormat);
		B_file.close();
	}
    std::ofstream C_file(C_file_path);
	if (C_file.is_open()) {
		C_file << this->C.format(CSVFormat);
		C_file.close();
	}
    std::ofstream x0_file(x0_file_path);
	if (x0_file.is_open()) {
		x0_file << this->x0.format(CSVFormat);
		x0_file.close();
	}
    std::ofstream input_sequence_file(input_sequence_file_path);
	if (input_sequence_file.is_open()) {
		input_sequence_file << this->input_sequence.format(CSVFormat);
		input_sequence_file.close();
	}
    std::ofstream output_sequence_file(output_sequence_file_path);
	if (output_sequence_file.is_open()) {
		output_sequence_file << this->output_sequence.format(CSVFormat);
		output_sequence_file.close();
	}
}

/**
 * Generate an Eigen matrix out of a previously populated file from the save_data() function
*/
Eigen::MatrixXd SUV::open_data(std::string file_path, bool is_output /*=false*/) {
	
	// the input is the file: "fileToOpen.csv":
	// a,b,c
	// d,e,f
	// This function converts input file data into the Eigen matrix format

	// the matrix entries are stored in this variable row-wise. For example if we have the matrix:
	// M=[a b c 
	//	  d e f]
	// the entries are stored as matrixEntries=[a,b,c,d,e,f], that is the variable "matrixEntries" is a row vector
	// later on, this vector is mapped into the Eigen matrix format
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
    this->simulation_ended = is_output;
	// here we conver the vector variable into the matrix and return the resulting object, 
	// note that matrixEntries.data() is the pointer to the first memory location at which the entries of the vector matrixEntries are stored;
	return Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> (matrixEntries.data(), matrixRowNumber, matrixEntries.size() / matrixRowNumber);
}

/**
 * After having initialized the SUV use this function to populate it with the data stored from a file.
*/
void SUV::open_from_files(std::string A_file_path, std::string B_file_path,
                    std::string C_file_path, std::string x0_file_path,
                    std::string input_sequence_file_path) {

	this->A = SUV::open_data(A_file_path);
	this->B = SUV::open_data(B_file_path);
	this->C = SUV::open_data(C_file_path);
	this->x0 = SUV::open_data(x0_file_path);
	this->input_sequence = this->open_data(input_sequence_file_path);

	this->n = this->A.rows();
	this->m = this->B.cols();
	this->r = this->C.rows();
	this->N = this->input_sequence.cols();

	this->output_sequence.resize(this->r, this->N); this->output_sequence.setZero();
	this->state_sequence.resize(this->n, this->N);	this->state_sequence.setZero();

	this->time_row.resize(1, this->N);

	for (int i = 0; i < this->N; i++)
	{
		this->time_row(0, i) = i + 1;
	}
    this->simulation_ended = false;
}

/**
 * Get state sequence, output sequence and time row vector from the SUV.
*/
std::tuple<Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd> SUV::get_state_output_time() {
	std::tuple<Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd> result(this->state_sequence, this->output_sequence, this->time_row);
	return result;
}

/**
 * Run one step and return its outpu.
 * Only use for testing and debugging purposes.
*/
Eigen::MatrixXd SUV::run_one_step(Eigen::MatrixXd u, bool store_history/*=true*/, bool verbose/*=false*/) {
    // u must be of size (m,1)
    if (store_history) {
        this->state_sequence.col(this->current_time) = this->A*this->old_x + this->B*u;
        this->x = this->state_sequence.col(this->current_time);
        this->output_sequence.col(this->current_time) = this->C*this->x;
        this->y = this->output_sequence.col(this->current_time);
    } else {
        this->x = this->A*this->old_x + this->B*u;
        this->y = this->C*this->x;

        this->old_x = this->x;
    }
    this->current_time++;
    return this->y;
}