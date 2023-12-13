// Real-Time Nonparametric Anomaly Detection

#include "include/RTNAD.h"

#include <iostream>
#include <assert.h>

using namespace std;


void RTNAD::update_decision_statistic() {
    cout << "Updating decison statistic." << endl;
}
void RTNAD::chech_anomaly() {
    if (this->g >= h) {
        this->anomaly = true;
    } else if ((this->g >= 0) && (this->g < this->h)){
        this->anomaly = false;
    } else {
        assert(this->g >= 0);
    }
}