#include "include/statistics.h"

#include <iostream>

Stats::Stats() {
    this->mean = 0.0;
}
void Stats::compute_mean() {
    std::cout << "Computing mean." << std::endl;
}
void Stats::compute_summary_statistics() {
    std::cout << "Computing summary statistics." << std::endl;
}
void Stats::compute_tail_probability() {
    std::cout << "Computing tail probability p_t and s_t." << std::endl;
}