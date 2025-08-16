// hq_rssi.c
#include "hq_rssi.h"
#include <math.h>

float rssi_estimate_power(const int8_t* iq, size_t n) {
    if (!iq || n == 0) return -120.0f;
    
    double sum = 0.0;
    for (size_t i = 0; i < n * 2; i += 2) {
        float i_val = (float)iq[i] / 128.0f;
        float q_val = (float)iq[i+1] / 128.0f;
        sum += i_val * i_val + q_val * q_val;
    }
    
    double mean_power = sum / (double)n;
    if (mean_power <= 0.0) return -120.0f;
    
    // Convert to dBm with rough calibration
    float dbm = 10.0f * log10f(mean_power) + 10.0f;
    
    // Clamp to reasonable range
    if (dbm < -120.0f) dbm = -120.0f;
    if (dbm > 0.0f) dbm = 0.0f;
    
    return dbm;
}

float rssi_apply_ema(float prev, float now, float alpha) {
    if (alpha <= 0.0f || alpha > 1.0f) alpha = 0.25f;
    
    // Handle first measurement
    if (prev <= -120.0f) return now;
    
    return alpha * now + (1.0f - alpha) * prev;
}