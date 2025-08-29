"""
Централизованная обработка ошибок для ПАНОРАМА RSSI.
"""

import logging
import traceback
from typing import Optional, Callable, Any
from functools import wraps
from PyQt5.QtWidgets import QMessageBox, QWidget


class ErrorHandler:
    """Централизованный обработчик ошибок."""
    
    def __init__(self, logger: Optional[logging.Logger] = None, parent: Optional[QWidget] = None):
        self.log = logger or logging.getLogger(__name__)
        self.parent = parent
    
    def handle_error(self, error: Exception, context: str = "", show_dialog: bool = False) -> None:
        """Обрабатывает ошибку с логированием и опциональным диалогом."""
        error_msg = f"{context}: {str(error)}" if context else str(error)
        self.log.error(error_msg)
        self.log.debug(f"Error traceback: {traceback.format_exc()}")
        
        if show_dialog and self.parent:
            QMessageBox.critical(
                self.parent,
                "Ошибка",
                f"Произошла ошибка: {error_msg}"
            )
    
    def handle_warning(self, message: str, show_dialog: bool = False) -> None:
        """Обрабатывает предупреждение."""
        self.log.warning(message)
        
        if show_dialog and self.parent:
            QMessageBox.warning(
                self.parent,
                "Предупреждение",
                message
            )
    
    def safe_execute(self, func: Callable, *args, context: str = "", 
                    default_return: Any = None, show_error: bool = False, **kwargs) -> Any:
        """Безопасно выполняет функцию с обработкой ошибок."""
        try:
            return func(*args, **kwargs)
        except Exception as e:
            self.handle_error(e, context, show_error)
            return default_return


def safe_method(context: str = "", show_error: bool = False, default_return: Any = None):
    """Декоратор для безопасного выполнения методов с обработкой ошибок."""
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            try:
                return func(self, *args, **kwargs)
            except Exception as e:
                # Пытаемся найти логгер и обработчик ошибок в объекте
                if hasattr(self, 'error_handler'):
                    self.error_handler.handle_error(e, context or f"{func.__name__}", show_error)
                elif hasattr(self, 'log'):
                    self.log.error(f"{context or func.__name__}: {str(e)}")
                    self.log.debug(f"Error traceback: {traceback.format_exc()}")
                else:
                    # Fallback логирование
                    logger = logging.getLogger(func.__module__)
                    logger.error(f"{context or func.__name__}: {str(e)}")
                
                return default_return
        return wrapper
    return decorator


def create_error_handler_for_class(cls):
    """Создает обработчик ошибок для класса и добавляет его как атрибут."""
    def init_wrapper(original_init):
        @wraps(original_init)
        def wrapper(self, *args, **kwargs):
            original_init(self, *args, **kwargs)
            
            # Создаем обработчик ошибок
            logger = getattr(self, 'log', None)
            parent = self if hasattr(self, 'parent') else None
            self.error_handler = ErrorHandler(logger, parent)
        return wrapper
    
    cls.__init__ = init_wrapper(cls.__init__)
    return cls