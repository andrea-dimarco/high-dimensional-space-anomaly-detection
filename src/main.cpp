// Main

#include "include/suv.h"
#include "include/gem.h"

#include <iostream>
#include <random>
#include <cstring>
#include <cstdlib>
#include <assert.h>

#include <Eigen/Dense>

/**
 * Main function of our SUV.
 */
int main()
{
    int p = 1; // output dimension
    // sensors are not independent within eachother at time t
    // different samples taken ad different times t and t' are i.i.d.

    int tau = 10; // change-point (when the anomaly begins)

    // the model is unkown so must be simulated as i.i.d. variables
    // decide what significant statistics need to be taken

    int N = 5; // number of samples in the nominal data set (data guaranteed to have no anomalies)

    GEM gem(p);

    return 0;
} /* main */
