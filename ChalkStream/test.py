import sys
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QPushButton,
    QLabel,
    QVBoxLayout,
    QHBoxLayout
)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Right Side Button Panel")

        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Main horizontal layout
        main_layout = QHBoxLayout()
        central_widget.setLayout(main_layout)

        # --------------------
        # LEFT: Main content
        # --------------------
        content_widget = QWidget()
        content_layout = QVBoxLayout()
        content_widget.setLayout(content_layout)

        self.label = QLabel("Main Content Area")
        content_layout.addWidget(self.label)

        # --------------------
        # RIGHT: Button panel
        # --------------------
        button_panel = QWidget()
        button_layout = QVBoxLayout()
        button_panel.setLayout(button_layout)

        btn1 = QPushButton("Action 1")
        btn2 = QPushButton("Action 2")

        btn1.clicked.connect(self.action1)
        btn2.clicked.connect(self.action2)

        button_layout.addWidget(btn1)
        button_layout.addWidget(btn2)
        button_layout.addStretch()  # pushes buttons to top

        # Add both sections to main layout
        main_layout.addWidget(content_widget, 5)  # stretch factor 3
        main_layout.addWidget(button_panel, 1)    # stretch factor 1

    def action1(self):
        self.label.setText("Action 1 triggered")

    def action2(self):
        self.label.setText("Action 2 triggered")


app = QApplication(sys.argv)
window = MainWindow()
window.resize(800, 400)
window.show()
sys.exit(app.exec())