#pragma once

class RTNAD {
private:
    int t;        // time-step
    double p;
    double s;
    double alpha;
    double g;     // decision statistic
    double h;     // decision threshold
    bool   anomaly;
    
public:
    void update_decision_statistic();
    void chech_anomaly();
};