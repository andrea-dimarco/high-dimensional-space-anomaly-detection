#pragma once

class Stats {
private:
    float mean;
public:
    Stats();
    void compute_mean();
    void compute_summary_statistics();
    void compute_tail_probability();
};