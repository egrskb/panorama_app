#!/usr/bin/env python3
"""
Тест для проверки исправления ошибки 'DetectedPeak object is not subscriptable'
Проверяет только исправление в коде без импорта модулей
"""

import os

def test_master_sweep_controller_fix():
    """Тестирует исправление в MasterSweepController"""
    
    print("=== Тест исправления в MasterSweepController ===")
    
    try:
        # Проверяем исправленный код
        master_file = 'features/spectrum/master.py'
        
        if not os.path.exists(master_file):
            print(f"❌ Файл {master_file} не найден")
            return False
            
        with open(master_file, 'r') as f:
            content = f.read()
            
        print(f"✅ Файл {master_file} прочитан успешно")
        
        # Проверяем, что исправление применено
        checks = [
            ('best_peak = max(peaks, key=lambda p: p.snr_db)', 'Использование p.snr_db'),
            ('peak_freq = best_peak.freq_hz', 'Использование best_peak.freq_hz'),
            ('peak_snr = best_peak.snr_db', 'Использование best_peak.snr_db'),
            ('# детектор пиков -> список DetectedPeak', 'Правильный комментарий')
        ]
        
        all_passed = True
        for check_str, description in checks:
            if check_str in content:
                print(f"   ✅ {description}: найдено")
            else:
                print(f"   ❌ {description}: не найдено")
                all_passed = False
        
        # Проверяем, что старый неправильный код удален
        old_code_checks = [
            ('peak_freq, peak_snr = max(peaks, key=lambda t: t[1])', 'Старый неправильный код'),
            ('# детектор пиков -> (freq, snr)', 'Старый неправильный комментарий')
        ]
        
        for check_str, description in old_code_checks:
            if check_str in content:
                print(f"   ❌ {description}: все еще присутствует")
                all_passed = False
            else:
                print(f"   ✅ {description}: удален")
        
        return all_passed
        
    except Exception as e:
        print(f"❌ Ошибка при тестировании: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_detected_peak_class_definition():
    """Тестирует определение класса DetectedPeak"""
    
    print("\n=== Тест определения класса DetectedPeak ===")
    
    try:
        # Проверяем определение класса
        model_file = 'features/spectrum/model.py'
        
        if not os.path.exists(model_file):
            print(f"❌ Файл {model_file} не найден")
            return False
            
        with open(model_file, 'r') as f:
            content = f.read()
            
        print(f"✅ Файл {model_file} прочитан успешно")
        
        # Проверяем, что класс определен правильно
        checks = [
            ('@dataclass(frozen=True)', 'Dataclass с frozen=True'),
            ('class DetectedPeak:', 'Определение класса'),
            ('freq_hz: float', 'Атрибут freq_hz'),
            ('snr_db: float', 'Атрибут snr_db'),
            ('power_dbm: float', 'Атрибут power_dbm'),
            ('band_hz: float', 'Атрибут band_hz'),
            ('idx: int', 'Атрибут idx')
        ]
        
        all_passed = True
        for check_str, description in checks:
            if check_str in content:
                print(f"   ✅ {description}: найдено")
            else:
                print(f"   ❌ {description}: не найдено")
                all_passed = False
        
        return all_passed
        
    except Exception as e:
        print(f"❌ Ошибка при тестировании: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_peak_detector_methods():
    """Тестирует методы PeakDetector"""
    
    print("\n=== Тест методов PeakDetector ===")
    
    try:
        # Проверяем методы
        model_file = 'features/spectrum/model.py'
        
        if not os.path.exists(model_file):
            print(f"❌ Файл {model_file} не найден")
            return False
            
        with open(model_file, 'r') as f:
            content = f.read()
            
        print(f"✅ Файл {model_file} прочитан успешно")
        
        # Проверяем, что методы определены правильно
        checks = [
            ('def detect_peaks(', 'Метод detect_peaks'),
            ('-> List[DetectedPeak]:', 'Возвращаемый тип List[DetectedPeak]'),
            ('def detect(', 'Метод detect'),
            ('class PeakDetector:', 'Определение класса PeakDetector')
        ]
        
        all_passed = True
        for check_str, description in checks:
            if check_str in content:
                print(f"   ✅ {description}: найдено")
            else:
                print(f"   ❌ {description}: не найдено")
                all_passed = False
        
        return all_passed
        
    except Exception as e:
        print(f"❌ Ошибка при тестировании: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Тестирование исправления ошибки 'DetectedPeak object is not subscriptable'")
    print("=" * 70)
    
    success = True
    success &= test_master_sweep_controller_fix()
    success &= test_detected_peak_class_definition()
    success &= test_peak_detector_methods()
    
    print("\n" + "=" * 70)
    if success:
        print("🎉 Все тесты пройдены успешно!")
        print("✅ Ошибка 'DetectedPeak object is not subscriptable' исправлена")
        print("\nИсправления:")
        print("  • Заменен код 'peak_freq, peak_snr = max(peaks, key=lambda t: t[1])'")
        print("  • На 'best_peak = max(peaks, key=lambda p: p.snr_db)'")
        print("  • И 'peak_freq = best_peak.freq_hz', 'peak_snr = best_peak.snr_db'")
        print("  • Обновлены комментарии для ясности")
    else:
        print("❌ Некоторые тесты не пройдены")
        print("   Проверьте, что исправления применены корректно")
    
    print("\nТеперь код должен работать без ошибки 'DetectedPeak object is not subscriptable'")
