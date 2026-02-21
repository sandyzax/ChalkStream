# gui_app.py
import sys
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QThread
from worker import Worker
from main_window import MainWindow

class GuiApp:
    def __init__(self, controller):
        self.app = QApplication(sys.argv)
        self.win = MainWindow(1000, 800)

        self.thread = QThread()
        self.worker = Worker()

        self.worker.moveToThread(self.thread)

        controller.start_work.connect(self.worker.load_images)
        self.worker.image_ready.connect(self.win.load_image)

        self.thread.start()
        self.win.show()

    def run(self):
        sys.exit(self.app.exec())
