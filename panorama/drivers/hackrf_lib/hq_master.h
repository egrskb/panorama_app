// hq_master.h
#ifndef HQ_MASTER_H
#define HQ_MASTER_H

#include "hq_init.h"

// Максимальное количество точек спектра
#define MAX_SPECTRUM_POINTS 50000

// External globals
extern double g_grouping_tolerance_hz;

typedef struct {
    double start_hz;
    double stop_hz;
    double step_hz;
    uint32_t dwell_ms;
} SweepPlan;

int start_master(SdrCtx* master, const SweepPlan* plan);
void stop_master(SdrCtx* master);

#endif // HQ_MASTER_H