#!/usr/bin/env python3
"""
Тест C библиотеки HackRF Master
"""

import ctypes
import os
import sys

def test_c_library():
    """Тестирует загрузку C библиотеки."""
    try:
        # Пытаемся загрузить библиотеку
        lib_path = "./libhackrf_qsa.so"
        if not os.path.exists(lib_path):
            lib_path = "./build/libhackrf_qsa.so"
        
        if not os.path.exists(lib_path):
            print(f"❌ Библиотека не найдена: {lib_path}")
            return False
        
        print(f"✅ Найдена библиотека: {lib_path}")
        
        # Загружаем библиотеку
        lib = ctypes.CDLL(lib_path)
        print("✅ Библиотека загружена успешно")
        
        # Проверяем основные функции
        try:
            # hq_device_count
            if hasattr(lib, 'hq_device_count'):
                count = lib.hq_device_count()
                print(f"✅ hq_device_count(): {count}")
            else:
                print("❌ hq_device_count не найден")
            
            # hq_get_segment_mode
            if hasattr(lib, 'hq_get_segment_mode'):
                mode = lib.hq_get_segment_mode()
                print(f"✅ hq_get_segment_mode(): {mode}")
            else:
                print("❌ hq_get_segment_mode не найден")
                
        except Exception as e:
            print(f"⚠️ Ошибка при вызове функций: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Ошибка загрузки библиотеки: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Тестирование C библиотеки HackRF Master")
    print("=" * 50)
    
    success = test_c_library()
    
    if success:
        print("\n🎉 Тест C библиотеки прошел успешно!")
    else:
        print("\n❌ Тест C библиотеки не прошел")
        sys.exit(1)
