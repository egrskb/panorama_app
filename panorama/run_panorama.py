#!/usr/bin/env python3
"""
Скрипт запуска PANORAMA из корня проекта
"""

import sys
import os

# Добавляем корень проекта в PYTHONPATH
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Импортируем и запускаем
from panorama.main import main

if __name__ == "__main__":
    main()