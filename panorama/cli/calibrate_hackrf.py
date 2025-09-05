#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CLI команда для калибровки HackRF устройств
Использование: python -m panorama.cli.calibrate_hackrf [опции]
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import List, Optional

# Импорт модулей проекта
try:
    from panorama.features.calibration import HackRFSyncCalibrator, CalibrationTarget
    from panorama.drivers.hackrf.hackrf_slaves.hackrf_slave_wrapper import HackRFSlaveDevice
except ImportError as e:
    print(f"Ошибка импорта: {e}")
    print("Убедитесь, что вы находитесь в корне проекта и все зависимости установлены")
    sys.exit(1)


def setup_logging(level: str = "INFO") -> logging.Logger:
    """Настройка логирования."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger(__name__)


def discover_hackrf_devices() -> List[str]:
    """Поиск всех доступных HackRF устройств."""
    devices = []
    
    try:
        # Попробуем открыть устройства с пустыми серийниками для перечисления
        for i in range(10):  # Максимум 10 устройств
            try:
                device = HackRFSlaveDevice(serial="", logger=logging.getLogger("discovery"))
                if device.open():
                    # Здесь должен быть способ получить серийный номер
                    # Пока используем индекс как идентификатор
                    devices.append(f"hackrf_{i}")
                    device.close()
                else:
                    break
            except Exception:
                break
    except Exception as e:
        logging.getLogger(__name__).warning(f"Ошибка поиска устройств: {e}")
    
    return devices


def calibrate_single_device(serial: str, logger: logging.Logger) -> bool:
    """Калибровка одного устройства."""
    try:
        calibrator = HackRFSyncCalibrator(logger=logger)
        
        # Создаем master и slave устройства
        master_device = HackRFSlaveDevice(serial="", logger=logger)  # Первое устройство как master
        slave_device = HackRFSlaveDevice(serial=serial, logger=logger)
        
        if not master_device.open():
            logger.error("Не удалось открыть master устройство")
            return False
        
        if not slave_device.open():
            logger.error(f"Не удалось открыть slave устройство {serial}")
            master_device.close()
            return False
        
        logger.info(f"Начало калибровки устройства {serial}")
        
        # Выполняем калибровку
        success = calibrator.calibrate_device_pair(master_device, slave_device, serial)
        
        # Закрываем устройства
        slave_device.close()
        master_device.close()
        
        if success:
            logger.info(f"Калибровка {serial} завершена успешно")
        else:
            logger.error(f"Калибровка {serial} не удалась")
        
        return success
        
    except Exception as e:
        logger.error(f"Ошибка калибровки {serial}: {e}")
        return False


def calibrate_all_devices(logger: logging.Logger) -> bool:
    """Калибровка всех доступных устройств."""
    devices = discover_hackrf_devices()
    
    if len(devices) < 2:
        logger.error(f"Недостаточно устройств для калибровки (найдено {len(devices)}, нужно минимум 2)")
        return False
    
    logger.info(f"Найдено {len(devices)} устройств HackRF")
    
    master_serial = devices[0]  # Первое устройство - master
    slave_serials = devices[1:]  # Остальные - slaves
    
    success_count = 0
    
    for slave_serial in slave_serials:
        logger.info(f"Калибровка {slave_serial} относительно {master_serial}")
        
        if calibrate_single_device(slave_serial, logger):
            success_count += 1
        
        time.sleep(1)  # Пауза между устройствами
    
    logger.info(f"Калибровка завершена: {success_count}/{len(slave_serials)} устройств")
    return success_count > 0


def show_calibration_status(logger: logging.Logger) -> None:
    """Показать статус калибровки всех устройств."""
    try:
        calibrator = HackRFSyncCalibrator(logger=logger)
        devices = discover_hackrf_devices()
        
        if not devices:
            logger.info("HackRF устройства не найдены")
            return
        
        print(f"\n{'='*70}")
        print(f"{'СТАТУС КАЛИБРОВКИ HACKRF УСТРОЙСТВ':^70}")
        print(f"{'='*70}")
        print(f"{'Устройство':<15} {'Частотн. смещ.':<15} {'Амплитуд. смещ.':<15} {'Возраст':<15} {'Статус':<10}")
        print(f"{'-'*70}")
        
        for device_serial in devices:
            info = calibrator.get_calibration_info(device_serial)
            
            if info:
                freq_offset = f"{info['frequency_offset_hz']:.0f} Гц"
                amp_offset = f"{info['amplitude_offset_db']:.1f} дБ" 
                age = f"{info['age_hours']:.1f} ч"
                status = "OK" if info['age_hours'] < 24 else "УСТАРЕЛА"
            else:
                freq_offset = "НЕТ ДАННЫХ"
                amp_offset = "НЕТ ДАННЫХ"
                age = "НЕТ ДАННЫХ"
                status = "НЕ КАЛИБР."
            
            print(f"{device_serial:<15} {freq_offset:<15} {amp_offset:<15} {age:<15} {status:<10}")
        
        print(f"{'='*70}\n")
        
    except Exception as e:
        logger.error(f"Ошибка получения статуса калибровки: {e}")


def main():
    """Главная функция CLI."""
    parser = argparse.ArgumentParser(
        description="Калибровка HackRF устройств для синхронизации частот и амплитуд",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:
  
  # Показать статус калибровки всех устройств
  python -m panorama.cli.calibrate_hackrf --status
  
  # Калибровать все доступные устройства
  python -m panorama.cli.calibrate_hackrf --calibrate-all
  
  # Калибровать конкретное устройство
  python -m panorama.cli.calibrate_hackrf --device hackrf_1234
  
  # Калибровка с подробным логированием
  python -m panorama.cli.calibrate_hackrf --calibrate-all --log-level DEBUG
        """
    )
    
    parser.add_argument(
        "--status", "-s",
        action="store_true",
        help="Показать статус калибровки всех устройств"
    )
    
    parser.add_argument(
        "--calibrate-all", "-a",
        action="store_true", 
        help="Калибровать все доступные HackRF устройства"
    )
    
    parser.add_argument(
        "--device", "-d",
        type=str,
        metavar="SERIAL",
        help="Калибровать конкретное устройство по серийному номеру"
    )
    
    parser.add_argument(
        "--log-level", "-l",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Уровень логирования (по умолчанию: INFO)"
    )
    
    parser.add_argument(
        "--config-path", "-c",
        type=Path,
        help="Путь к файлу конфигурации калибровки"
    )
    
    args = parser.parse_args()
    
    # Настройка логирования
    logger = setup_logging(args.log_level)
    
    # Если не указаны опции - показываем помощь
    if not any([args.status, args.calibrate_all, args.device]):
        parser.print_help()
        return 0
    
    try:
        # Показать статус калибровки
        if args.status:
            show_calibration_status(logger)
        
        # Калибровать все устройства
        elif args.calibrate_all:
            logger.info("Начало калибровки всех HackRF устройств")
            success = calibrate_all_devices(logger)
            if success:
                logger.info("Калибровка завершена успешно")
                show_calibration_status(logger)
                return 0
            else:
                logger.error("Калибровка не удалась")
                return 1
        
        # Калибровать конкретное устройство
        elif args.device:
            logger.info(f"Начало калибровки устройства {args.device}")
            success = calibrate_single_device(args.device, logger)
            if success:
                logger.info("Калибровка завершена успешно")
                return 0
            else:
                logger.error("Калибровка не удалась")
                return 1
    
    except KeyboardInterrupt:
        logger.info("Калибровка прервана пользователем")
        return 1
    except Exception as e:
        logger.error(f"Критическая ошибка: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())