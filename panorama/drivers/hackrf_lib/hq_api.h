#ifndef HQ_API_H
#define HQ_API_H

#include <stdint.h>
#include <stdbool.h>
#include <errno.h>

// Публичные структуры для FFI
typedef struct {
    double f_center_hz;
    double bw_hz;
    float rssi_ema;
    uint64_t last_ns;
    int hit_count;
} WatchItem;

typedef struct {
    double f_hz;
    float rssi_dbm;
    uint64_t last_ns;
} Peak;

typedef struct {
    int master_running;
    int slave_running[2];
    double retune_ms_avg;
    int watch_items;
} HqStatus;

// API функции
int  hq_open_all(int num_expected);
void hq_close_all(void);

int  hq_config_set_rates(uint32_t samp_rate_hz, uint32_t bb_bw_hz);
int  hq_config_set_gains(uint32_t lna_db, uint32_t vga_db, bool amp_on);

int  hq_start(void);
void hq_stop(void);

int  hq_get_watchlist_snapshot(WatchItem* out, int max_items);
int  hq_get_recent_peaks(Peak* out, int max_items);

void hq_set_grouping_tolerance_hz(double delta_hz);
void hq_set_ema_alpha(float alpha);

int  hq_get_status(HqStatus* out);

#endif // HQ_API_H