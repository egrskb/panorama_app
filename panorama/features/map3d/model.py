from __future__ import annotations
from dataclasses import dataclass

@dataclass
class Node:
    name: str
    x: float
    y: float
    color: str  # hex

class MapModel:
    def __init__(self):
        self.master = Node("Master", 0.0, 0.0, "#1f77b4")   # синий
        self.slave1 = Node("Slave 1", 10.0, 0.0, "#d62728") # красный
        self.slave2 = Node("Slave 2", 0.0, 10.0, "#2ca02c") # зелёный
