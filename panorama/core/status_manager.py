"""
Менеджер статуса системы ПАНОРАМА RSSI.
"""

from typing import Dict, Any, Callable
from PyQt5.QtCore import QObject, pyqtSignal, QTimer


class SystemStatusManager(QObject):
    """Менеджер для отслеживания и управления статусом системы."""
    
    status_updated = pyqtSignal(dict)
    
    def __init__(self, parent=None, update_interval_ms: int = 1000):
        super().__init__(parent)
        
        self.status = {
            'master_running': False,
            'orchestrator_running': False,
            'n_slaves': 0,
            'n_targets': 0
        }
        
        self._status_callbacks: Dict[str, Callable] = {}
        
        # Таймер для периодического обновления
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self._update_status)
        self.update_timer.start(update_interval_ms)
    
    def set_status(self, key: str, value: Any) -> None:
        """Устанавливает значение статуса."""
        if self.status.get(key) != value:
            self.status[key] = value
            self.status_updated.emit(self.status.copy())
    
    def get_status(self, key: str = None) -> Any:
        """Получает значение статуса."""
        if key is None:
            return self.status.copy()
        return self.status.get(key)
    
    def register_status_callback(self, status_key: str, callback: Callable) -> None:
        """Регистрирует callback для получения статуса компонента."""
        self._status_callbacks[status_key] = callback
    
    def _update_status(self) -> None:
        """Обновляет статус системы."""
        try:
            # Обновляем статус через зарегистрированные callbacks
            updated = False
            
            for key, callback in self._status_callbacks.items():
                try:
                    new_value = callback()
                    if self.status.get(key) != new_value:
                        self.status[key] = new_value
                        updated = True
                except Exception:
                    pass
            
            if updated:
                self.status_updated.emit(self.status.copy())
                
        except Exception:
            pass
    
    def format_status_title(self, base_title: str = "ПАНОРАМА RSSI") -> str:
        """Форматирует заголовок окна с текущим статусом."""
        title = f"{base_title} - Master: {'ON' if self.status['master_running'] else 'OFF'}, "
        title += f"Orch: {'ON' if self.status['orchestrator_running'] else 'OFF'}, "
        title += f"Slaves: {self.status['n_slaves']}, Targets: {self.status['n_targets']}"
        return title
    
    def stop(self) -> None:
        """Останавливает менеджер статуса."""
        if self.update_timer.isActive():
            self.update_timer.stop()