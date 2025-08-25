#!/usr/bin/env python3
"""
Тест для проверки исправления ошибки 'DetectedPeak object is not subscriptable'
"""

import numpy as np
import sys
import os

# Добавляем путь к модулям
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from panorama.features.spectrum.model import PeakDetector, DetectedPeak

def test_peak_detection():
    """Тестирует детекцию пиков и убеждается, что возвращаются объекты DetectedPeak"""
    
    print("=== Тест детекции пиков ===")
    
    # Создаем тестовые данные
    freqs = np.linspace(100e6, 6000e6, 1000)  # 100 МГц - 6 ГГц
    power = np.random.normal(-100, 5, 1000)  # Шум -100 дБм ±5 дБ
    
    # Добавляем несколько пиков
    peak_indices = [200, 400, 600, 800]
    for idx in peak_indices:
        power[idx] = -50  # Сильный сигнал -50 дБм
    
    # Создаем детектор
    detector = PeakDetector(
        snr_threshold_db=10.0,
        min_peak_bins=3,
        min_peak_distance_bins=5,
        peak_band_hz=5e6
    )
    
    # Детектируем пики
    peaks = detector.detect_peaks(freqs, power)
    
    print(f"Найдено пиков: {len(peaks)}")
    
    # Проверяем, что возвращаются объекты DetectedPeak
    for i, peak in enumerate(peaks):
        print(f"Пик {i+1}:")
        print(f"  Тип: {type(peak)}")
        print(f"  Частота: {peak.freq_hz/1e6:.2f} МГц")
        print(f"  SNR: {peak.snr_db:.1f} дБ")
        print(f"  Мощность: {peak.power_dbm:.1f} дБм")
        print(f"  Индекс: {peak.idx}")
        
        # Проверяем, что это действительно объект DetectedPeak
        assert isinstance(peak, DetectedPeak), f"Ожидался DetectedPeak, получен {type(peak)}"
        
        # Проверяем, что можно обращаться к атрибутам
        assert hasattr(peak, 'freq_hz'), "Отсутствует атрибут freq_hz"
        assert hasattr(peak, 'snr_db'), "Отсутствует атрибут snr_db"
        assert hasattr(peak, 'power_dbm'), "Отсутствует атрибут power_dbm"
        assert hasattr(peak, 'idx'), "Отсутствует атрибут idx"
    
    # Тестируем поиск максимального пика по SNR
    if peaks:
        best_peak = max(peaks, key=lambda p: p.snr_db)
        print(f"\nЛучший пик по SNR:")
        print(f"  Частота: {best_peak.freq_hz/1e6:.2f} МГц")
        print(f"  SNR: {best_peak.snr_db:.1f} дБ")
        
        # Проверяем, что можно извлечь значения
        peak_freq = best_peak.freq_hz
        peak_snr = best_peak.snr_db
        
        print(f"  Извлеченные значения: freq={peak_freq/1e6:.2f} МГц, snr={peak_snr:.1f} дБ")
    
    print("\n✅ Тест пройден успешно!")
    return True

def test_peak_attributes():
    """Тестирует создание и атрибуты объекта DetectedPeak"""
    
    print("\n=== Тест атрибутов DetectedPeak ===")
    
    # Создаем объект DetectedPeak
    peak = DetectedPeak(
        freq_hz=2400e6,      # 2.4 ГГц
        snr_db=25.0,         # SNR 25 дБ
        power_dbm=-45.0,     # Мощность -45 дБм
        band_hz=5e6,         # Полоса 5 МГц
        idx=500               # Индекс 500
    )
    
    print(f"Создан пик: {peak}")
    print(f"Тип: {type(peak)}")
    print(f"Атрибуты:")
    print(f"  freq_hz: {peak.freq_hz} ({peak.freq_hz/1e6:.1f} МГц)")
    print(f"  snr_db: {peak.snr_db}")
    print(f"  power_dbm: {peak.power_dbm}")
    print(f"  band_hz: {peak.band_hz} ({peak.band_hz/1e6:.1f} МГц)")
    print(f"  idx: {peak.idx}")
    
    # Проверяем, что объект неизменяемый (frozen dataclass)
    try:
        peak.freq_hz = 5000e6
        print("❌ Ошибка: объект должен быть неизменяемым")
        return False
    except Exception as e:
        print(f"✅ Объект неизменяемый (как и должно быть): {e}")
    
    print("✅ Тест атрибутов пройден успешно!")
    return True

if __name__ == "__main__":
    try:
        success = True
        success &= test_peak_detection()
        success &= test_peak_attributes()
        
        if success:
            print("\n🎉 Все тесты пройдены успешно!")
            sys.exit(0)
        else:
            print("\n❌ Некоторые тесты не пройдены")
            sys.exit(1)
            
    except Exception as e:
        print(f"\n💥 Ошибка в тестах: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
