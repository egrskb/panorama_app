#!/usr/bin/env python3
"""
Скрипт запуска ПАНОРАМА RSSI - системы трилатерации по RSSI.
"""

import sys
import os
import logging
from pathlib import Path

# ОТКЛЮЧАЕМ AVAHI В SOAPYSDR ДО ВСЕХ ИМПОРТОВ
# Это предотвращает ошибки "avahi_service_browser_new() failed: Bad state"
os.environ['SOAPY_SDR_DISABLE_AVAHI'] = '1'

# Добавляем корневую директорию в путь
sys.path.insert(0, str(Path(__file__).parent))

def setup_environment():
    """Настраивает окружение для запуска."""
    
    # Проверяем наличие необходимых переменных окружения
    if 'PYTHONPATH' not in os.environ:
        os.environ['PYTHONPATH'] = str(Path(__file__).parent)
    
    # Проверяем наличие необходимых модулей
    try:
        import PyQt5
        print("✓ PyQt5 доступен")
    except ImportError:
        print("✗ PyQt5 не найден. Установите: pip install PyQt5")
        return False
    
    try:
        import numpy
        print("✓ NumPy доступен")
    except ImportError:
        print("✗ NumPy не найден. Установите: pip install numpy")
        return False
    
    try:
        import scipy
        print("✓ SciPy доступен")
    except ImportError:
        print("✗ SciPy не найден. Установите: pip install scipy")
        return False
    
    # Проверяем наличие наших модулей
    try:
       from panorama.features.spectrum.master import MasterSweepController
       print("✓ Модуль Master доступен")
    except ImportError as e:
        print(f"✗ Модуль Master не найден: {e}")
        return False
    
    try:
        from panorama.features.slave_sdr.slave import SlaveManager
        print("✓ Модуль Slave доступен")
    except ImportError as e:
        print(f"✗ Модуль Slave не найден: {e}")
        return False
    
    try:
        from panorama.features.trilateration import RSSITrilaterationEngine
        print("✓ Модуль трилатерации доступен")
    except ImportError as e:
        print(f"✗ Модуль трилатерации не найден: {e}")
        return False
    
    try:
        from panorama.features.orchestrator.core import Orchestrator
        print("✓ Модуль оркестратора доступен")
    except ImportError as e:
        print(f"✗ Модуль оркестратора не найден: {e}")
        return False
    
    try:
        from panorama.features.calibration.manager import CalibrationManager
        print("✓ Модуль калибровки доступен")
    except ImportError as e:
        print(f"✗ Модуль калибровки не найден: {e}")
        return False
    
    return True

def check_dependencies():
    """Проверяет зависимости для работы с SDR."""
    print("\nПроверка зависимостей SDR:")
    
    # Проверяем SoapySDR
    try:
        import SoapySDR
        print("✓ SoapySDR доступен")
    except ImportError:
        print("✗ SoapySDR не найден. Установите: pip install SoapySDR")
        print("  Примечание: для работы с SDR устройствами")
    
    # Проверяем CFFI-библиотеку HackRF Master
    try:
        from panorama.drivers.hrf_backend import HackRFQSABackend
        print("✓ HackRF Master (CFFI) доступен")
    except Exception as e:
        print(f"✗ HackRF Master (CFFI) не найден: {e}")
        print("  Подсказка: соберите библиотеку: ./build_hackrf_master.sh")
    
    # Проверяем существующие модули
    try:
        from panorama.features.map import OpenLayersMapWidget
        print("✓ OpenLayersMapWidget доступен")
    except ImportError:
        print("✗ OpenLayersMapWidget не найден")
        print("  Примечание: для отображения карты")
    
    try:
        from panorama.features.spectrum import SpectrumView
        print("✓ SpectrumView доступен")
    except ImportError:
        print("✗ SpectrumView не найден")
        print("  Примечание: для отображения спектра")

def main():
    """Главная функция запуска."""
    print("ПАНОРАМА RSSI - Система трилатерации по RSSI")
    print("=" * 50)
    
    # Настраиваем логирование
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Проверяем окружение
    if not setup_environment():
        print("\n❌ Окружение не настроено. Исправьте ошибки и попробуйте снова.")
        return 1
    
    # Проверяем зависимости
    check_dependencies()
    
    print("\n🚀 Запуск приложения...")
    
    try:
        # Импортируем и запускаем главное приложение
        from panorama.main_rssi import main as app_main
        app_main()
        return 0
        
    except ImportError as e:
        print(f"\n❌ Ошибка импорта: {e}")
        print("Проверьте, что все модули установлены корректно.")
        return 1
        
    except Exception as e:
        print(f"\n❌ Ошибка запуска: {e}")
        logging.exception("Unexpected error during startup")
        return 1

if __name__ == "__main__":
    sys.exit(main())
