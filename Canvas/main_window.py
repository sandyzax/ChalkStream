# main_window.py
from PySide6.QtWidgets import QMainWindow, QLabel


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("GUI Example")
        self.resize(400, 200)

        self.label = QLabel("Waiting...", self)
        self.label.setGeometry(50, 80, 300, 40)

    def load_image(self, text):
        print("GUI received:", text)
        self.label.setText(text)
