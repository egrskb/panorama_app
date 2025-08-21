#!/usr/bin/env python3
"""
Простой тест CFFI загрузки HackRF Master библиотеки
"""

from cffi import FFI
from pathlib import Path

def test_cffi_load():
    """Тестирует загрузку библиотеки через CFFI."""
    
    HERE = Path(__file__).parent / "panorama/drivers/hackrf_master"
    LIB_PATH = HERE / "build" / "libhackrf_master.so"
    HEADER_PATH = HERE / "hackrf_master.h"
    
    print(f"LIB_PATH: {LIB_PATH}")
    print(f"HEADER_PATH: {HEADER_PATH}")
    print(f"LIB exists: {LIB_PATH.exists()}")
    print(f"HEADER exists: {HEADER_PATH.exists()}")
    
    if not LIB_PATH.exists():
        print("❌ Библиотека не найдена")
        return False
        
    if not HEADER_PATH.exists():
        print("❌ Заголовок не найден")
        return False
    
    try:
        # Загружаем заголовок
        header_content = HEADER_PATH.read_text(encoding="utf-8")
        print(f"Заголовок загружен, размер: {len(header_content)} символов")
        
        # Создаем FFI
        ffi = FFI()
        
        # Очищаем заголовок от проблемных строк
        lines = []
        for line in header_content.splitlines():
            line = line.strip()
            if line.startswith('#'):
                continue
            if 'extern "C"' in line:
                continue
            if line == '{' or line == '}':
                continue
            lines.append(line)
        
        clean_header = "\n".join(lines)
        print(f"Очищенный заголовок, размер: {len(clean_header)} символов")
        
        # Пытаемся загрузить
        print("Пытаемся загрузить через cdef...")
        ffi.cdef(clean_header, override=True)
        print("✓ cdef успешен")
        
        # Пытаемся загрузить библиотеку
        print("Пытаемся загрузить библиотеку...")
        lib = ffi.dlopen(str(LIB_PATH))
        print("✓ dlopen успешен")
        
        # Пытаемся вызвать функцию
        print("Пытаемся вызвать hackrf_master_init...")
        result = lib.hackrf_master_init()
        print(f"✓ hackrf_master_init вернул: {result}")
        
        # Очистка
        lib.hackrf_master_cleanup()
        print("✓ cleanup успешен")
        
        return True
        
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=== Тест CFFI загрузки HackRF Master ===")
    success = test_cffi_load()
    if success:
        print("\n✅ Тест прошел успешно!")
    else:
        print("\n❌ Тест провалился!")
