#!/usr/bin/env python3
"""
Тестовый скрипт для проверки multi-HackRF режима
"""

import time
import numpy as np
from cffi import FFI

def test_multi_hackrf():
    """Тестирует multi-HackRF режим."""
    
    # Создаем FFI
    ffi = FFI()
    
    # Определяем интерфейс
    ffi.cdef("""
        // Структуры данных для multi-SDR
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
        
        // Multi-SDR API
        int  hq_open_all(int num_expected);
        void hq_close_all(void);
        
        int  hq_config_set_rates(uint32_t samp_rate_hz, uint32_t bb_bw_hz);
        int  hq_config_set_gains(uint32_t lna_db, uint32_t vga_db, bool amp_on);
        
        // Настройка диапазона частот
        int  hq_config_set_freq_range(double start_hz, double stop_hz, double step_hz);
        int  hq_config_set_dwell_time(uint32_t dwell_ms);
        
        int  hq_start(void);
        void hq_stop(void);
        
        int  hq_get_watchlist_snapshot(WatchItem* out, int max_items);
        int  hq_get_recent_peaks(Peak* out, int max_items);
        
        // Чтение непрерывного спектра от Master SDR
        int  hq_get_master_spectrum(double* freqs_hz, float* powers_dbm, int max_points);
        
        void hq_set_grouping_tolerance_hz(double delta_hz);
        void hq_set_ema_alpha(float alpha);
        
        int  hq_get_status(HqStatus* out);
        
        // Device enumeration
        int  hq_list_devices(char* serials[], int max_count);
        int  hq_get_device_count(void);
    """)
    
    try:
        # Загружаем библиотеку
        lib = ffi.dlopen("./libhackrf_multi.so")
        print("✓ Библиотека загружена успешно")
        
        # Проверяем количество устройств
        device_count = lib.hq_get_device_count()
        print(f"✓ Доступно устройств: {device_count}")
        
        # Открываем устройства
        print("Открываем устройства...")
        result = lib.hq_open_all(1)  # Только master для теста
        if result != 0:
            print(f"❌ Ошибка открытия устройств: {result}")
            return
        
        print("✓ Устройства открыты")
        
        # Настраиваем параметры
        print("Настраиваем параметры...")
        lib.hq_config_set_rates(12000000, 8000000)  # 12 MSPS, 8 MHz BB filter
        lib.hq_config_set_gains(24, 20, False)  # LNA: 24dB, VGA: 20dB, AMP: off
        
        # Настраиваем диапазон частот (2.4-2.5 GHz)
        start_hz = 2.4e9
        stop_hz = 2.5e9
        step_hz = 1e6  # 1 MHz шаг
        
        lib.hq_config_set_freq_range(start_hz, stop_hz, step_hz)
        lib.hq_config_set_dwell_time(2)  # 2 ms dwell time
        
        # Настраиваем параметры группировки
        lib.hq_set_grouping_tolerance_hz(250000.0)  # 250 kHz
        lib.hq_set_ema_alpha(0.25)
        
        print(f"✓ Диапазон настроен: {start_hz/1e9:.1f}-{stop_hz/1e9:.1f} GHz, шаг {step_hz/1e6:.1f} MHz")
        
        # Запускаем
        print("Запускаем multi-SDR...")
        result = lib.hq_start()
        if result != 0:
            print(f"❌ Ошибка запуска: {result}")
            lib.hq_close_all()
            return
        
        print("✓ Multi-SDR запущен")
        
        # Буферы для чтения данных
        max_points = 1000
        freqs_buf = ffi.new("double[]", max_points)
        powers_buf = ffi.new("float[]", max_points)
        status_buf = ffi.new("HqStatus*")
        
        # Основной цикл тестирования
        print("Начинаем тестирование...")
        start_time = time.time()
        
        for i in range(10):  # 10 итераций
            time.sleep(0.5)  # Ждем 500ms
            
            # Читаем статус
            lib.hq_get_status(status_buf)
            status = status_buf[0]
            
            print(f"Итерация {i+1}: Master={status.master_running}, "
                  f"Slave1={status.slave_running[0]}, Slave2={status.slave_running[1]}, "
                  f"Targets={status.watch_items}")
            
            # Читаем спектр
            n_read = lib.hq_get_master_spectrum(freqs_buf, powers_buf, max_points)
            
            if n_read > 0:
                # Копируем данные
                freqs_array = np.frombuffer(
                    ffi.buffer(freqs_buf, n_read * 8),
                    dtype=np.float64,
                )
                powers_array = np.frombuffer(
                    ffi.buffer(powers_buf, n_read * 4),
                    dtype=np.float32,
                )
                
                # Анализируем данные
                active_points = np.sum(powers_array > -110)
                max_power = np.max(powers_array)
                min_power = np.min(powers_array)
                
                print(f"  Спектр: {n_read} точек, активных: {active_points}, "
                      f"мощность: {min_power:.1f}...{max_power:.1f} dBm")
                
                # Ищем пики
                if active_points > 0:
                    peak_indices = np.where(powers_array > -80)[0]
                    if len(peak_indices) > 0:
                        print(f"  Найдены пики на частотах:")
                        for idx in peak_indices[:5]:  # Показываем первые 5
                            freq_mhz = freqs_array[idx] / 1e6
                            power = powers_array[idx]
                            print(f"    {freq_mhz:.1f} MHz: {power:.1f} dBm")
            else:
                print(f"  Спектр не готов (n_read={n_read})")
        
        elapsed = time.time() - start_time
        print(f"✓ Тестирование завершено за {elapsed:.1f} секунд")
        
        # Останавливаем
        print("Останавливаем multi-SDR...")
        lib.hq_stop()
        lib.hq_close_all()
        print("✓ Multi-SDR остановлен")
        
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_multi_hackrf()
