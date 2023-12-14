#pragma once

#include <iostream>
#include <tuple>
#include <string>
#include <fstream>
#include <vector>
#include <Eigen/Dense>
#include <Eigen/src/Core/Map.h>

// x(t) = A*x(t-1) + B*u(t-1)
// y(t) = C*x(t-1)

class SUV {
private:
    int n; // state dimension
    int m; // input dimension
    int r; // output dimension
    int N; // simulation time
    int current_time;
    Eigen::MatrixXd x0; // initial state (vector)
    Eigen::MatrixXd A;  // transition matrix
    Eigen::MatrixXd x;  // current state (vector)
    Eigen::MatrixXd old_x; // past state (vector)
    Eigen::MatrixXd B;  // input processing
    Eigen::MatrixXd y;  // output (vector)
    Eigen::MatrixXd C;  // state processing

    Eigen::MatrixXd input_sequence;  // sequence of inputs
    Eigen::MatrixXd output_sequence;  // sequence of outputs
    Eigen::MatrixXd state_sequence;  // sequence of touched states

    Eigen::MatrixXd time_row; // time row (vector)

    bool simulation_ended;

public:

    SUV();
    SUV(Eigen::MatrixXd Am, Eigen::MatrixXd Bm,
        Eigen::MatrixXd Cm, Eigen::MatrixXd initial_state,
        Eigen::MatrixXd input_sequence_m);

    void save_data(std::string A_file_path, std::string B_file_path,
                    std::string C_file_path, std::string x0_file_path,
                    std::string input_sequence_file_path, std::string output_sequence_file_path);

    Eigen::MatrixXd open_data(std::string file_path, bool is_output=false);

    void open_from_files(std::string A_file_path, std::string B_file_path,
                    std::string C_file_path, std::string x0_file_path,
                    std::string input_sequence_file_path);

    void run_simulation(bool store_history=true, bool verbose=false);

    void run_simulation_up_to(int t, bool store_history=true, bool verbose=false);

	std::tuple<Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd> get_state_output_time();

    Eigen::MatrixXd get_output_at_time(int t, bool store_history=true, bool verbose=false);

    Eigen::MatrixXd run_one_step(Eigen::MatrixXd input, bool store_history=true, bool verbose=false);
};