# controller.py
from PySide6.QtCore import QObject, Signal

class Controller(QObject):
    start_work = Signal()
