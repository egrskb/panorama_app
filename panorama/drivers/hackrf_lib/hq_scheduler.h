// hq_scheduler.h
#ifndef HQ_SCHEDULER_H
#define HQ_SCHEDULER_H

#include "hq_grouping.h"

typedef struct {
    int slave_idx;
    double center_hz;
    WatchItem* targets[MAX_WATCHLIST];
    int num_targets;
} SlavePlan;

void scheduler_assign_targets(WatchItem* watchlist, size_t n_items, 
                             SlavePlan* plans, int n_slaves);

#endif // HQ_SCHEDULER_H