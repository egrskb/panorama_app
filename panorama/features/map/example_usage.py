"""
Пример использования модульной системы карт.
Демонстрирует как интегрировать новую систему карт в существующее приложение.
"""

import sys
from typing import Optional
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QPushButton, QHBoxLayout
from PyQt5.QtCore import QTimer
import random

from .manager import MapManager
from .config import MapConfig, RSSIConfig, TrilatationConfig


class ExampleMapWindow(QMainWindow):
    """Пример окна с интегрированной картой трилатерации."""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("RSSI Трилатерация - Демонстрация")
        self.setGeometry(100, 100, 1200, 800)
        
        # Создаем конфигурацию
        self.map_config = self._create_config()
        
        # Создаем менеджер карты
        self.map_manager = MapManager(self.map_config, self)
        
        # Создаем UI
        self._setup_ui()
        
        # Подключаем сигналы
        self._connect_signals()
        
        # Демонстрационные данные
        self._setup_demo_data()
    
    def _create_config(self) -> MapConfig:
        """Создает конфигурацию карты."""
        config = MapConfig.create_default()
        
        # Настройки RSSI
        config.rssi.threshold_dbm = -75.0
        config.rssi.visualization_radius_m = 75
        config.rssi.update_rate_hz = 10
        config.rssi.color_scheme = "traffic_light"
        
        # Настройки трилатерации
        config.trilateration.confidence_threshold = 0.6
        config.trilateration.track_drones = True
        config.trilateration.show_trajectories = True
        
        # Добавляем станции для демонстрации
        config.add_station("slave1", 150.0, 100.0, 0.0, "Станция 1")
        config.add_station("slave2", -100.0, 150.0, 0.0, "Станция 2")  
        config.add_station("slave3", 50.0, -120.0, 0.0, "Станция 3")
        
        return config
    
    def _setup_ui(self):
        """Создает пользовательский интерфейс."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        layout = QVBoxLayout(central_widget)
        
        # Панель управления
        controls_layout = QHBoxLayout()
        
        self.btn_add_drone = QPushButton("Добавить дрон")
        self.btn_add_drone.clicked.connect(self._add_random_drone)
        controls_layout.addWidget(self.btn_add_drone)
        
        self.btn_add_rssi = QPushButton("Добавить RSSI")
        self.btn_add_rssi.clicked.connect(self._add_random_rssi)
        controls_layout.addWidget(self.btn_add_rssi)
        
        self.btn_clear = QPushButton("Очистить")
        self.btn_clear.clicked.connect(self._clear_all)
        controls_layout.addWidget(self.btn_clear)
        
        self.btn_center = QPushButton("Центрировать")
        self.btn_center.clicked.connect(self._center_map)
        controls_layout.addWidget(self.btn_center)
        
        controls_layout.addStretch()
        layout.addLayout(controls_layout)
        
        # Карта
        self.map_widget = self.map_manager.initialize_widget(self)
        layout.addWidget(self.map_widget)
    
    def _connect_signals(self):
        """Подключает сигналы."""
        self.map_manager.map_ready.connect(self._on_map_ready)
        self.map_manager.drone_detected.connect(self._on_drone_detected)
        
        # Регистрируем callbacks
        self.map_manager.register_callback('drone_selected', self._on_drone_selected)
        self.map_manager.register_callback('station_selected', self._on_station_selected)
    
    def _setup_demo_data(self):
        """Настраивает демонстрационные данные."""
        # Таймер для автоматического обновления демо-данных
        self.demo_timer = QTimer()
        self.demo_timer.timeout.connect(self._update_demo_data)
        self.demo_timer.start(2000)  # Каждые 2 секунды
        
        self.demo_drone_counter = 0
    
    def _on_map_ready(self):
        """Обработчик готовности карты."""
        print("Карта готова к использованию!")
        
        # Добавляем начальные демо-данные
        self._add_initial_demo_data()
    
    def _add_initial_demo_data(self):
        """Добавляет начальные демо-данные."""
        # Добавляем дрон в центре
        self.map_manager.add_drone(
            "demo_drone_1", 
            0.0, 0.0,
            freq_mhz=433.5,
            rssi_dbm=-65.0,
            confidence=0.8,
            is_tracked=True
        )
        
        # Добавляем RSSI измерения вокруг станций
        stations = [("slave1", 150.0, 100.0), ("slave2", -100.0, 150.0), ("slave3", 50.0, -120.0)]
        for station_id, x, y in stations:
            # Добавляем RSSI измерения с убывающей мощностью от станции
            for radius in [50, 100, 150]:
                rssi = -60 - (radius / 10)  # Убывание сигнала с расстоянием
                self.map_manager.add_rssi_measurement(
                    x + random.uniform(-20, 20),
                    y + random.uniform(-20, 20),
                    rssi,
                    station_id,
                    433.5
                )
    
    def _update_demo_data(self):
        """Обновляет демо-данные."""
        # Двигаем существующие дроны
        for i in range(1, 3):
            drone_id = f"demo_drone_{i}"
            
            # Случайное движение
            x = random.uniform(-200, 200)
            y = random.uniform(-200, 200)
            rssi = random.uniform(-90, -50)
            confidence = random.uniform(0.3, 0.9)
            
            self.map_manager.add_drone(
                drone_id,
                x, y,
                freq_mhz=433.5,
                rssi_dbm=rssi,
                confidence=confidence,
                is_tracked=confidence > 0.7
            )
        
        # Добавляем случайные RSSI измерения
        for _ in range(3):
            x = random.uniform(-300, 300)
            y = random.uniform(-300, 300)
            rssi = random.uniform(-100, -50)
            
            self.map_manager.add_rssi_measurement(x, y, rssi, "demo_station", 433.5)
    
    def _add_random_drone(self):
        """Добавляет случайный дрон."""
        self.demo_drone_counter += 1
        drone_id = f"manual_drone_{self.demo_drone_counter}"
        
        x = random.uniform(-250, 250)
        y = random.uniform(-250, 250)
        rssi = random.uniform(-90, -40)
        confidence = random.uniform(0.2, 0.95)
        freq = random.choice([433.5, 868.0, 915.0, 2400.0])
        
        self.map_manager.add_drone(
            drone_id, x, y,
            freq_mhz=freq,
            rssi_dbm=rssi,
            confidence=confidence,
            is_tracked=confidence > 0.6
        )
        
        print(f"Добавлен дрон {drone_id}: ({x:.1f}, {y:.1f}), RSSI={rssi:.1f}dBm")
    
    def _add_random_rssi(self):
        """Добавляет случайное RSSI измерение."""
        x = random.uniform(-300, 300)
        y = random.uniform(-300, 300)
        rssi = random.uniform(-110, -40)
        freq = random.choice([433.5, 868.0, 915.0])
        
        self.map_manager.add_rssi_measurement(x, y, rssi, "manual_station", freq)
        
        print(f"Добавлено RSSI измерение: ({x:.1f}, {y:.1f}), {rssi:.1f}dBm")
    
    def _clear_all(self):
        """Очищает все данные."""
        self.map_manager.clear_all_drones()
        self.map_manager.data_manager.clear_rssi_measurements()
        print("Все данные очищены")
    
    def _center_map(self):
        """Центрирует карту."""
        self.map_manager.center_on_origin()
        print("Карта отцентрирована")
    
    def _on_drone_detected(self, drone_id: str, x: float, y: float, confidence: float):
        """Обработчик обнаружения дрона."""
        print(f"Обнаружен дрон {drone_id} в позиции ({x:.1f}, {y:.1f}) с уверенностью {confidence:.2f}")
    
    def _on_drone_selected(self, drone_id: str):
        """Обработчик выбора дрона."""
        print(f"Выбран дрон: {drone_id}")
    
    def _on_station_selected(self, station_id: str):
        """Обработчик выбора станции."""
        print(f"Выбрана станция: {station_id}")
    
    def closeEvent(self, event):
        """Обработчик закрытия окна."""
        if hasattr(self, 'demo_timer'):
            self.demo_timer.stop()
        print("Статистика карты:", self.map_manager.get_statistics())
        event.accept()


def main():
    """Главная функция демонстрации."""
    # WSL/без GPU: отключаем аппаратное ускорение
    import os
    os.environ.setdefault("QT_OPENGL", "software")
    os.environ.setdefault("QTWEBENGINE_DISABLE_GPU", "1")
    
    app = QApplication(sys.argv)
    
    window = ExampleMapWindow()
    window.show()
    
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()