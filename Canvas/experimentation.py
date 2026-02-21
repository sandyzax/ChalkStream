import threading
import sys
from PySide6.QtWidgets import QApplication, QWidget
import time


class MyApp:

    def __init__(self):
        self.app = QApplication(sys.argv)
        self.window = QWidget()
        self.window.show()

    def run(self):
        self.app.exec()

    def test(self):
        print(1)


def start_gui(app_obj):
    app_obj.run()


gui = MyApp()

t = threading.Thread(target=start_gui, args=(gui,), daemon=True)
t.start()

# Other code runs here
gui.test()
while True:
    time.sleep(1)
    gui.test()
