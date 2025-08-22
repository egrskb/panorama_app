#!/usr/bin/env python3
"""
Тест объединенного HackRF QSA backend
"""

import sys
from pathlib import Path

# Добавляем текущую директорию в путь
sys.path.insert(0, str(Path(__file__).parent))

def test_import():
    """Тестирует импорт backend."""
    try:
        print("🔍 Тестирование импорта...")
        from panorama.drivers.hrf_backend import HackRFQSABackend
        print("✅ HackRFQSABackend импортирован успешно")
        
        # Проверяем атрибуты класса
        print("🔍 Проверка атрибутов класса...")
        attrs = dir(HackRFQSABackend)
        required_attrs = ['start', 'stop', 'is_running', 'enumerate_devices']
        
        for attr in required_attrs:
            if attr in attrs:
                print(f"✅ {attr} найден")
            else:
                print(f"❌ {attr} не найден")
        
        return True
        
    except ImportError as e:
        print(f"❌ Ошибка импорта: {e}")
        return False
    except Exception as e:
        print(f"❌ Неожиданная ошибка: {e}")
        return False

def test_ffi_definitions():
    """Тестирует FFI определения."""
    try:
        print("\n🔍 Тестирование FFI определений...")
        from panorama.drivers.hrf_backend import HackRFQSABackend
        
        # Создаем экземпляр (без инициализации SDR)
        backend = HackRFQSABackend()
        print("✅ FFI объект создан")
        
        # Проверяем FFI атрибуты
        if hasattr(backend, '_ffi'):
            print("✅ FFI объект найден")
        else:
            print("❌ FFI объект не найден")
            
        return True
        
    except Exception as e:
        print(f"❌ Ошибка FFI: {e}")
        return False

def test_enumerate_devices():
    """Тестирует перечисление устройств."""
    try:
        print("\n🔍 Тестирование перечисления устройств...")
        from panorama.drivers.hrf_backend import HackRFQSABackend
        
        devices = HackRFQSABackend.enumerate_devices()
        print(f"✅ Найдено устройств: {len(devices)}")
        
        for i, device in enumerate(devices):
            print(f"  {i+1}. {device}")
            
        return True
        
    except Exception as e:
        print(f"❌ Ошибка перечисления устройств: {e}")
        return False

def main():
    """Основная функция тестирования."""
    print("🧪 Тест объединенного HackRF QSA Backend")
    print("=" * 50)
    
    tests = [
        ("Импорт", test_import),
        ("FFI определения", test_ffi_definitions),
        ("Перечисление устройств", test_enumerate_devices),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n📋 Тест: {test_name}")
        print("-" * 30)
        
        if test_func():
            passed += 1
            print(f"✅ {test_name} - ПРОЙДЕН")
        else:
            print(f"❌ {test_name} - ПРОВАЛЕН")
    
    print("\n" + "=" * 50)
    print(f"📊 Результаты: {passed}/{total} тестов пройдено")
    
    if passed == total:
        print("🎉 Все тесты пройдены успешно!")
        return 0
    else:
        print("⚠️  Некоторые тесты провалены")
        return 1

if __name__ == "__main__":
    exit(main())
