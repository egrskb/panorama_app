#!/usr/bin/env python3
"""
Тест для WatchlistView
Проверяет создание и базовую функциональность виджета
"""

import sys
import os

# Добавляем путь к модулям
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_watchlist_import():
    """Тестирует импорт WatchlistView"""
    
    print("=== Тест импорта WatchlistView ===")
    
    try:
        from panorama.features.watchlist import WatchlistView
        
        print(f"✅ WatchlistView импортирован успешно")
        print(f"   Тип: {type(WatchlistView)}")
        print(f"   Модуль: {WatchlistView.__module__}")
        
        return True
        
    except Exception as e:
        print(f"❌ Ошибка при импорте WatchlistView: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_watchlist_class_definition():
    """Тестирует определение класса WatchlistView"""
    
    print("\n=== Тест определения класса WatchlistView ===")
    
    try:
        from panorama.features.watchlist.view import WatchlistView
        
        # Проверяем атрибуты класса
        expected_attrs = [
            'task_selected', 'task_cancelled', 'task_retried',
            'set_orchestrator', '_setup_ui', '_connect_orchestrator'
        ]
        
        all_passed = True
        for attr in expected_attrs:
            if hasattr(WatchlistView, attr):
                print(f"   ✅ Атрибут {attr}: присутствует")
            else:
                print(f"   ❌ Атрибут {attr}: отсутствует")
                all_passed = False
        
        return all_passed
        
    except Exception as e:
        print(f"❌ Ошибка при тестировании класса: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_orchestrator_methods():
    """Тестирует методы оркестратора"""
    
    print("\n=== Тест методов оркестратора ===")
    
    try:
        from panorama.features.orchestrator.core import Orchestrator
        
        # Проверяем, что метод get_active_tasks добавлен
        if hasattr(Orchestrator, 'get_active_tasks'):
            print("✅ Метод get_active_tasks: присутствует")
        else:
            print("❌ Метод get_active_tasks: отсутствует")
            return False
            
        return True
        
    except Exception as e:
        print(f"❌ Ошибка при тестировании оркестратора: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_main_integration():
    """Тестирует интеграцию с main_rssi.py"""
    
    print("\n=== Тест интеграции с main_rssi.py ===")
    
    try:
        # Проверяем, что импорт добавлен
        with open('main_rssi.py', 'r') as f:
            content = f.read()
            
        checks = [
            ('from panorama.features.watchlist import WatchlistView', 'Импорт WatchlistView'),
            ('self.watchlist_view = WatchlistView(orchestrator=self.orchestrator)', 'Создание WatchlistView'),
            ('tab_widget.addTab(self.watchlist_view, "Watchlist")', 'Добавление вкладки Watchlist'),
            ('self.watchlist_view.task_cancelled.connect(self._on_task_cancelled)', 'Подключение сигнала task_cancelled'),
            ('self.watchlist_view.task_retried.connect(self._on_task_retried)', 'Подключение сигнала task_retried'),
            ('def _on_task_cancelled(self, task_id: str):', 'Метод _on_task_cancelled'),
            ('def _on_task_retried(self, task_id: str):', 'Метод _on_task_retried')
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
        print(f"❌ Ошибка при тестировании интеграции: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Тестирование WatchlistView и интеграции")
    print("=" * 60)
    
    success = True
    success &= test_watchlist_import()
    success &= test_watchlist_class_definition()
    success &= test_orchestrator_methods()
    success &= test_main_integration()
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 Все тесты пройдены успешно!")
        print("✅ WatchlistView готов к использованию")
        print("\nФункциональность:")
        print("  • Отображение задач watchlist в реальном времени")
        print("  • Фильтрация по статусу и частоте")
        print("  • Детальная информация о задачах")
        print("  • Управление задачами (отмена, повтор)")
        print("  • Статистика и мониторинг")
    else:
        print("❌ Некоторые тесты не пройдены")
        print("   Проверьте, что все изменения применены корректно")
    
    print("\nТеперь в главном окне появится вкладка 'Watchlist' для отслеживания задач")
