// hq_scheduler.c
#include "hq_scheduler.h"
#include <math.h>
#include <string.h>
#include <stdlib.h>

// Helper structure for clustering
typedef struct {
    double center_hz;
    int count;
    int indices[MAX_WATCHLIST];
} Cluster;

static int compare_freq(const void* a, const void* b) {
    WatchItem* wa = *(WatchItem**)a;
    WatchItem* wb = *(WatchItem**)b;
    
    if (wa->f_center_hz < wb->f_center_hz) return -1;
    if (wa->f_center_hz > wb->f_center_hz) return 1;
    return 0;
}

void scheduler_assign_targets(WatchItem* watchlist, size_t n_items, 
                             SlavePlan* plans, int n_slaves) {
    if (!watchlist || !plans || n_items == 0 || n_slaves <= 0) return;
    
    // Initialize plans
    for (int i = 0; i < n_slaves; i++) {
        plans[i].slave_idx = i;
        plans[i].center_hz = 0;
        plans[i].num_targets = 0;
    }
    
    // Create array of pointers for sorting
    WatchItem* sorted_items[MAX_WATCHLIST];
    size_t actual_count = n_items < MAX_WATCHLIST ? n_items : MAX_WATCHLIST;
    
    for (size_t i = 0; i < actual_count; i++) {
        sorted_items[i] = &watchlist[i];
    }
    
    // Sort by frequency
    qsort(sorted_items, actual_count, sizeof(WatchItem*), compare_freq);
    
    // Cluster targets that can be covered by single IF band (8 MHz)
    Cluster clusters[MAX_WATCHLIST];
    int n_clusters = 0;
    
    for (size_t i = 0; i < actual_count; i++) {
        int assigned = 0;
        
        // Try to add to existing cluster
        for (int c = 0; c < n_clusters; c++) {
            double center = clusters[c].center_hz;
            if (fabs(sorted_items[i]->f_center_hz - center) < 4e6) {
                // Add to this cluster
                clusters[c].indices[clusters[c].count++] = i;
                // Update center as weighted average
                clusters[c].center_hz = (center * (clusters[c].count - 1) + 
                                        sorted_items[i]->f_center_hz) / clusters[c].count;
                assigned = 1;
                break;
            }
        }
        
        // Create new cluster if not assigned
        if (!assigned && n_clusters < MAX_WATCHLIST) {
            clusters[n_clusters].center_hz = sorted_items[i]->f_center_hz;
            clusters[n_clusters].count = 1;
            clusters[n_clusters].indices[0] = i;
            n_clusters++;
        }
    }
    
    // Assign clusters to slaves (round-robin with load balancing)
    for (int c = 0; c < n_clusters; c++) {
        // Find slave with least targets
        int best_slave = 0;
        int min_targets = plans[0].num_targets;
        
        for (int s = 1; s < n_slaves; s++) {
            if (plans[s].num_targets < min_targets) {
                min_targets = plans[s].num_targets;
                best_slave = s;
            }
        }
        
        // Assign cluster to best slave
        SlavePlan* plan = &plans[best_slave];
        
        // Update center frequency (weighted average)
        if (plan->num_targets == 0) {
            plan->center_hz = clusters[c].center_hz;
        } else {
            plan->center_hz = (plan->center_hz * plan->num_targets + 
                              clusters[c].center_hz * clusters[c].count) / 
                             (plan->num_targets + clusters[c].count);
        }
        
        // Add targets from cluster
        for (int i = 0; i < clusters[c].count && plan->num_targets < MAX_WATCHLIST; i++) {
            plan->targets[plan->num_targets++] = sorted_items[clusters[c].indices[i]];
        }
    }
}