// hq_rssi.h
#ifndef HQ_RSSI_H
#define HQ_RSSI_H

#include <stddef.h>
#include <stdint.h>

float rssi_estimate_power(const int8_t* iq, size_t n);
float rssi_apply_ema(float prev, float now, float alpha);

#endif // HQ_RSSI_H