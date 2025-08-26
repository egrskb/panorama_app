#!/usr/bin/env python3
"""
Тест импорта HackRFQSABackend без PyQt5
"""

import sys
import os

# Добавляем путь к проекту
sys.path.insert(0, os.path.abspath('.'))

try:
    # Пытаемся импортировать только CFFI интерфейс
    from cffi import FFI
    print("✅ CFFI импортирован успешно")
    
    # Пытаемся импортировать backend
    from panorama.drivers.hrf_backend import HackRFQSABackend
    print("✅ HackRFQSABackend импортирован успешно")
    
    # Проверяем атрибуты класса
    print(f"Класс: {HackRFQSABackend}")
    print(f"Базовый класс: {HackRFQSABackend.__bases__}")
    
    # Проверяем методы
    methods = [attr for attr in dir(HackRFQSABackend) if not attr.startswith('_')]
    print(f"Публичные методы: {methods[:10]}...")  # Показываем первые 10
    
    print("\n🎉 Импорт прошел успешно!")
    
except ImportError as e:
    print(f"❌ Ошибка импорта: {e}")
    sys.exit(1)
except Exception as e:
    print(f"❌ Неожиданная ошибка: {e}")
    sys.exit(1)

