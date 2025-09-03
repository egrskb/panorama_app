# -*- coding: utf-8 -*-
"""
Модуль управления слейвами - объединяет watchlist, результаты и контроль оркестратора
в едином современном интерфейсе.
"""

from .view import ImprovedSlavesView
from .web_table_widget import WebTableWidget

__all__ = ['ImprovedSlavesView', 'WebTableWidget']
