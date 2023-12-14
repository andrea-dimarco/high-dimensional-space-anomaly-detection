#pragma once

class RTNAD {
private:
    int t;        // time-step
    float p;
    float s;
    float alpha;
    float g;     // decision statistic
    float h;     // decision threshold
    bool   anomaly;
    
public:
    void update_decision_statistic();
    void chech_anomaly();
};