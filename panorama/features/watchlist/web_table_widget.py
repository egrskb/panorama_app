"""
QWebEngineView виджет для отображения RSSI матрицы в веб-формате.
Обеспечивает двунаправленную связь между Python и JavaScript.
"""

import json
import logging
from pathlib import Path
from typing import Optional, Dict, List, Any
from PyQt5.QtCore import QUrl, pyqtSignal, QObject, pyqtSlot
from PyQt5.QtWebEngineWidgets import QWebEngineView, QWebEnginePage
from PyQt5.QtWebChannel import QWebChannel


class WebTableBridge(QObject):
    """Мост для связи между Python и JavaScript."""
    
    # Сигналы для передачи событий из веб-интерфейса
    export_requested = pyqtSignal(str)  # format
    map_navigate_requested = pyqtSignal(float, float)  # lat, lng
    clear_data_requested = pyqtSignal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.log = logging.getLogger(__name__)
    
    @pyqtSlot(str)
    def onExportRequested(self, format_name: str):
        """Обработка запроса на экспорт данных."""
        self.log.info(f"Export requested: {format_name}")
        self.export_requested.emit(format_name)
    
    @pyqtSlot(float, float)
    def onMapNavigate(self, lat: float, lng: float):
        """Обработка запроса на навигацию по карте."""
        self.log.info(f"Map navigate requested: {lat}, {lng}")
        self.map_navigate_requested.emit(lat, lng)
    
    @pyqtSlot()
    def onClearData(self):
        """Обработка запроса на очистку данных."""
        self.log.info("Clear data requested")
        self.clear_data_requested.emit()
    
    @pyqtSlot(str, result=str)
    def getConfiguration(self, key: str) -> str:
        """Получение конфигурации из Python."""
        # Возвращаем базовую конфигурацию
        config = {
            "theme": "light",
            "updateInterval": 1000,
            "colorScheme": "viridis",
            "showGrid": True,
            "showStats": True
        }
        return json.dumps(config.get(key, ""))


class WebTableWidget(QWebEngineView):
    """Виджет для отображения RSSI матрицы через QWebEngineView."""
    
    # Сигналы для интеграции с основным приложением
    export_requested = pyqtSignal(str)
    map_navigate_requested = pyqtSignal(float, float)
    clear_data_requested = pyqtSignal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.log = logging.getLogger(__name__)
        
        # Настройка веб-канала для связи с JavaScript
        self.bridge = WebTableBridge(self)
        self.channel = QWebChannel()
        self.channel.registerObject("pyBridge", self.bridge)
        self.page().setWebChannel(self.channel)
        
        # Подключение сигналов моста
        self.bridge.export_requested.connect(self.export_requested.emit)
        self.bridge.map_navigate_requested.connect(self.map_navigate_requested.emit)
        self.bridge.clear_data_requested.connect(self.clear_data_requested.emit)
        
        # Данные RSSI матрицы
        self._rssi_data: Dict[str, Dict[str, float]] = {}
        self._slaves_info: Dict[str, Dict[str, Any]] = {}
        self._targets_info: Dict[str, Dict[str, Any]] = {}
        
        # Загрузка HTML файла
        self._load_web_table()
        
        self.log.info("WebTableWidget initialized")
    
    def _load_web_table(self):
        """Загружает HTML файл веб-таблицы."""
        html_path = Path(__file__).parent / "web_table.html"
        if html_path.exists():
            url = QUrl.fromLocalFile(str(html_path.absolute()))
            self.load(url)
            self.log.info(f"Loaded web table from: {html_path}")
        else:
            self.log.error(f"Web table HTML not found: {html_path}")
            # Загружаем минимальный HTML как fallback
            self.setHtml(self._get_fallback_html())
    
    def _get_fallback_html(self) -> str:
        """Возвращает минимальный HTML в случае отсутствия основного файла."""
        return """
        <!DOCTYPE html>
        <html>
        <head>
            <title>RSSI Matrix</title>
            <style>
                body { font-family: Arial, sans-serif; padding: 20px; }
                .error { color: red; text-align: center; }
            </style>
        </head>
        <body>
            <div class="error">
                <h2>Ошибка загрузки веб-таблицы</h2>
                <p>Файл web_table.html не найден</p>
            </div>
        </body>
        </html>
        """
    
    def update_rssi_data(self, rssi_data: Dict[str, Dict[str, float]]):
        """Обновляет данные RSSI в веб-таблице."""
        self._rssi_data = rssi_data.copy()
        
        # Отправляем данные в JavaScript
        js_code = f"""
        if (typeof updateRSSIData === 'function') {{
            updateRSSIData({json.dumps(rssi_data)});
        }} else {{
            console.warn('updateRSSIData function not available');
        }}
        """
        self.page().runJavaScript(js_code)
        
        self.log.debug(f"Updated RSSI data: {len(rssi_data)} entries")
    
    def update_slaves_info(self, slaves_info: Dict[str, Dict[str, Any]]):
        """Обновляет информацию о слейвах."""
        self._slaves_info = slaves_info.copy()
        
        js_code = f"""
        if (typeof updateSlavesInfo === 'function') {{
            updateSlavesInfo({json.dumps(slaves_info)});
        }} else {{
            console.warn('updateSlavesInfo function not available');
        }}
        """
        self.page().runJavaScript(js_code)
        
        self.log.debug(f"Updated slaves info: {len(slaves_info)} slaves")
    
    def update_targets_info(self, targets_info: Dict[str, Dict[str, Any]]):
        """Обновляет информацию о целях."""
        self._targets_info = targets_info.copy()
        
        js_code = f"""
        if (typeof updateTargetsInfo === 'function') {{
            updateTargetsInfo({json.dumps(targets_info)});
        }} else {{
            console.warn('updateTargetsInfo function not available');
        }}
        """
        self.page().runJavaScript(js_code)
        
        self.log.debug(f"Updated targets info: {len(targets_info)} targets")
    
    def update_performance_stats(self, stats: Dict[str, Any]):
        """Обновляет статистику производительности."""
        js_code = f"""
        if (typeof updatePerformanceStats === 'function') {{
            updatePerformanceStats({json.dumps(stats)});
        }} else {{
            console.warn('updatePerformanceStats function not available');
        }}
        """
        self.page().runJavaScript(js_code)
    
    def set_theme(self, theme: str):
        """Устанавливает тему интерфейса."""
        js_code = f"""
        if (typeof setTheme === 'function') {{
            setTheme('{theme}');
        }} else {{
            console.warn('setTheme function not available');
        }}
        """
        self.page().runJavaScript(js_code)
        
        self.log.info(f"Set theme: {theme}")
    
    def set_color_scheme(self, scheme: str):
        """Устанавливает цветовую схему RSSI."""
        js_code = f"""
        if (typeof setColorScheme === 'function') {{
            setColorScheme('{scheme}');
        }} else {{
            console.warn('setColorScheme function not available');
        }}
        """
        self.page().runJavaScript(js_code)
        
        self.log.info(f"Set color scheme: {scheme}")
    
    def clear_all_data(self):
        """Очищает все данные в таблице."""
        self._rssi_data.clear()
        self._slaves_info.clear()
        self._targets_info.clear()
        
        js_code = """
        if (typeof clearAllData === 'function') {
            clearAllData();
        } else {
            console.warn('clearAllData function not available');
        }
        """
        self.page().runJavaScript(js_code)
        
        self.log.info("Cleared all data")
    
    def export_data(self, format_name: str) -> Dict[str, Any]:
        """Экспортирует данные в указанном формате."""
        export_data = {
            "timestamp": self._get_current_timestamp(),
            "rssi_data": self._rssi_data,
            "slaves_info": self._slaves_info,
            "targets_info": self._targets_info,
            "format": format_name
        }
        
        self.log.info(f"Exported data in format: {format_name}")
        return export_data
    
    def _get_current_timestamp(self) -> str:
        """Возвращает текущий timestamp."""
        from datetime import datetime
        return datetime.now().isoformat()
    
    def get_current_data(self) -> Dict[str, Any]:
        """Возвращает текущие данные таблицы."""
        return {
            "rssi_data": self._rssi_data,
            "slaves_info": self._slaves_info,
            "targets_info": self._targets_info
        }