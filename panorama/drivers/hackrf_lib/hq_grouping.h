// hq_grouping.h
#ifndef HQ_GROUPING_H
#define HQ_GROUPING_H

#include <stdint.h>
#include <stdbool.h>
#include <pthread.h>

#define MAX_WATCHLIST 100

typedef struct {
    double f_hz;
    float rssi_dbm;
    uint64_t last_ns;
} Peak;

typedef struct {
    double f_center_hz;
    double bw_hz;
    float rssi_ema;
    uint64_t last_ns;
    int hit_count;
} WatchItem;

typedef struct {
    Peak* buffer;
    size_t capacity;
    size_t head;
    size_t tail;
    pthread_mutex_t mutex;
} PeakQueue;

PeakQueue* peak_queue_create(size_t capacity);
void peak_queue_destroy(PeakQueue* q);
int peak_queue_push(PeakQueue* q, const Peak* peak);
int peak_queue_pop(PeakQueue* q, Peak* peak);
size_t peak_queue_size(PeakQueue* q);

void add_peak(double f_hz, float rssi_dbm);
void regroup_frequencies(double delta_hz);

// External globals
extern PeakQueue* g_peaks_queue;
extern WatchItem g_watchlist[];
extern size_t g_watchlist_count;
extern pthread_mutex_t g_watchlist_mutex;
extern float g_ema_alpha;

#endif // HQ_GROUPING_H