#!/usr/bin/env python3
"""
Простой тест для проверки исправления ошибки 'DetectedPeak object is not subscriptable'
Без зависимостей от PyQt5
"""

import sys
import os

# Добавляем путь к модулям
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_detected_peak_class():
    """Тестирует определение класса DetectedPeak"""
    
    print("=== Тест класса DetectedPeak ===")
    
    try:
        # Импортируем только базовые классы
        from panorama.features.spectrum.model import DetectedPeak
        
        print(f"✅ Класс DetectedPeak импортирован успешно")
        print(f"   Тип: {type(DetectedPeak)}")
        print(f"   Модуль: {DetectedPeak.__module__}")
        
        # Проверяем атрибуты класса
        expected_attrs = ['freq_hz', 'snr_db', 'power_dbm', 'band_hz', 'idx']
        for attr in expected_attrs:
            if hasattr(DetectedPeak, attr):
                print(f"   ✅ Атрибут {attr}: присутствует")
            else:
                print(f"   ❌ Атрибут {attr}: отсутствует")
        
        # Создаем тестовый объект
        peak = DetectedPeak(
            freq_hz=2400e6,      # 2.4 ГГц
            snr_db=25.0,         # SNR 25 дБ
            power_dbm=-45.0,     # Мощность -45 дБм
            band_hz=5e6,         # Полоса 5 МГц
            idx=500               # Индекс 500
        )
        
        print(f"✅ Объект DetectedPeak создан успешно")
        print(f"   Тип: {type(peak)}")
        print(f"   freq_hz: {peak.freq_hz}")
        print(f"   snr_db: {peak.snr_db}")
        print(f"   power_dbm: {peak.power_dbm}")
        
        return True
        
    except Exception as e:
        print(f"❌ Ошибка при тестировании DetectedPeak: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_peak_detector_import():
    """Тестирует импорт PeakDetector"""
    
    print("\n=== Тест импорта PeakDetector ===")
    
    try:
        from panorama.features.spectrum.model import PeakDetector
        
        print(f"✅ Класс PeakDetector импортирован успешно")
        print(f"   Тип: {type(PeakDetector)}")
        print(f"   Модуль: {PeakDetector.__module__}")
        
        return True
        
    except Exception as e:
        print(f"❌ Ошибка при импорте PeakDetector: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_master_sweep_controller():
    """Тестирует исправление в MasterSweepController"""
    
    print("\n=== Тест исправления в MasterSweepController ===")
    
    try:
        # Проверяем исправленный код
        from panorama.features.spectrum.master import MasterSweepController
        
        print(f"✅ MasterSweepController импортирован успешно")
        
        # Проверяем, что исправление применено
        with open('panorama/features/spectrum/master.py', 'r') as f:
            content = f.read()
            
        if 'best_peak = max(peaks, key=lambda p: p.snr_db)' in content:
            print("✅ Исправление применено: используется p.snr_db")
        else:
            print("❌ Исправление не найдено")
            return False
            
        if 'peak_freq = best_peak.freq_hz' in content:
            print("✅ Исправление применено: используется best_peak.freq_hz")
        else:
            print("❌ Исправление не найдено")
            return False
            
        if 'peak_snr = best_peak.snr_db' in content:
            print("✅ Исправление применено: используется best_peak.snr_db")
        else:
            print("❌ Исправление не найдено")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Ошибка при тестировании MasterSweepController: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Тестирование исправления ошибки 'DetectedPeak object is not subscriptable'")
    print("=" * 70)
    
    success = True
    success &= test_detected_peak_class()
    success &= test_peak_detector_import()
    success &= test_master_sweep_controller()
    
    print("\n" + "=" * 70)
    if success:
        print("🎉 Все тесты пройдены успешно!")
        print("✅ Ошибка 'DetectedPeak object is not subscriptable' исправлена")
        sys.exit(0)
    else:
        print("❌ Некоторые тесты не пройдены")
        sys.exit(1)
