# worker.py
from PySide6.QtCore import QObject, Signal, Slot
import time


class Worker(QObject):
    image_ready = Signal(str)

    @Slot()
    def load_images(self):
        print("Worker: loading images...")

        # Simulate heavy work
        time.sleep(2)

        self.image_ready.emit("Image 1 loaded")
        time.sleep(1)

        self.image_ready.emit("Image 2 loaded")

        print("Worker: done")
