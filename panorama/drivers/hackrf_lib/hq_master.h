// hq_master.h
#ifndef HQ_MASTER_H
#define HQ_MASTER_H

#include "hq_init.h"

typedef struct {
    double start_hz;
    double stop_hz;
    double step_hz;
    uint32_t dwell_ms;
} SweepPlan;

int start_master(SdrCtx* master, const SweepPlan* plan);
void stop_master(SdrCtx* master);

#endif // HQ_MASTER_H