# launcher.py
import threading
from controller import Controller
from gui_app import GuiApp

controller = Controller()

def start_gui():
    gui = GuiApp(controller)
    gui.run()

gui_thread = threading.Thread(target=start_gui, daemon=True)
gui_thread.start()

# Launcher can now emit signals at any time
print("Doing other work...")
controller.start_work.emit()
