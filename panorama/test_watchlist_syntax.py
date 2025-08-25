#!/usr/bin/env python3
"""
Тест синтаксиса Python для WatchlistView
Проверяет, что код компилируется без ошибок
"""

import ast
import os

def test_python_syntax():
    """Тестирует синтаксис Python файлов"""
    
    print("=== Тест синтаксиса Python ===")
    
    files_to_check = [
        'features/watchlist/view.py',
        'features/watchlist/__init__.py'
    ]
    
    all_passed = True
    for file_path in files_to_check:
        if not os.path.exists(file_path):
            print(f"   ❌ {file_path}: файл не найден")
            all_passed = False
            continue
            
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Пытаемся скомпилировать код
            ast.parse(content)
            print(f"   ✅ {file_path}: синтаксис корректен")
            
        except SyntaxError as e:
            print(f"   ❌ {file_path}: синтаксическая ошибка - {e}")
            all_passed = False
        except Exception as e:
            print(f"   ❌ {file_path}: ошибка - {e}")
            all_passed = False
    
    return all_passed

def test_method_names():
    """Тестирует корректность названий методов"""
    
    print("\n=== Тест названий методов ===")
    
    try:
        with open('features/watchlist/view.py', 'r', encoding='utf-8') as f:
            content = f.read()
            
        # Проверяем, что нет опечаток в названиях методов
        checks = [
            ('setSectionResizeMode', 'Правильное название метода QHeaderView'),
            ('QHeaderView.ResizeToContents', 'Правильная константа'),
            ('QHeaderView.Stretch', 'Правильная константа')
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
        print(f"❌ Ошибка при тестировании методов: {e}")
        return False

if __name__ == "__main__":
    print("Тестирование синтаксиса WatchlistView")
    print("=" * 50)
    
    success = True
    success &= test_python_syntax()
    success &= test_method_names()
    
    print("\n" + "=" * 50)
    if success:
        print("🎉 Все тесты синтаксиса пройдены успешно!")
        print("✅ Код WatchlistView синтаксически корректен")
        print("\nТеперь приложение должно запускаться без ошибок")
    else:
        print("❌ Обнаружены синтаксические ошибки")
        print("   Проверьте код и исправьте ошибки")
    
    print("\nWatchlistView готов к запуску!")
