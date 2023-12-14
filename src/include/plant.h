#pragma once

#include <Eigen/Dense>

class Plant {
private:
    int t; // time-step

public:
    Plant();
    void query();
};