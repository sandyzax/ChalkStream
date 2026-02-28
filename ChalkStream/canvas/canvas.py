import sys
import subprocess
from PySide6.QtCore import QPointF
from PySide6.QtWidgets import (
    QApplication,
    QGraphicsView,
    QGraphicsScene,
    QGraphicsPixmapItem,
    QMainWindow,
    QWidget,
    QPushButton,
    QLabel,
    QVBoxLayout,
    QHBoxLayout
)
import cv2
from PySide6.QtGui import QPainter
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtCore import Qt, QTimer

import time

class ImageItem(QGraphicsPixmapItem):

    def __init__(self, item_id, pixmap, scene):
        super().__init__(pixmap)

        self.scene_ref = scene
        self.item_id = item_id

        self.setFlags(
            QGraphicsPixmapItem.ItemIsMovable |
            QGraphicsPixmapItem.ItemIsSelectable |
            QGraphicsPixmapItem.ItemSendsGeometryChanges
        )


    def itemChange(self, change, value):

        if change == QGraphicsPixmapItem.ItemPositionChange and self.scene():

            new_pos = value

            scene_rect = self.scene().sceneRect()
            item_rect = self.boundingRect()

            left = scene_rect.left()
            top = scene_rect.top()
            right = scene_rect.right() - item_rect.width()
            bottom = scene_rect.bottom() - item_rect.height()

            x = min(max(new_pos.x(), left), right)
            y = min(max(new_pos.y(), top), bottom)

            return QPointF(x, y)

        return super().itemChange(change, value)
    
    def update_pixmap(self, new_pixmap):

        old_pos = self.pos()
        self.setPixmap(new_pixmap)
        self.setPos(old_pos)


def cv2_to_qpixmap(cv_img):
    """
    Convert OpenCV image (numpy array) to QPixmap
    """

    if cv_img is None:
        return QPixmap()

    rgb_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)

    h, w, ch = rgb_img.shape
    bytes_per_line = ch * w

    qimage = QImage(
        rgb_img.data,
        w,
        h,
        bytes_per_line,
        QImage.Format_RGB888
    )

    return QPixmap.fromImage(qimage)

class CanvasView(QGraphicsView):

    def __init__(self,dimension_x,dimension_y):
        super().__init__()

        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)
        self.scene.setSceneRect(0, 0, dimension_x, dimension_y)

        self.setRenderHint(QPainter.Antialiasing)
        self.setRenderHint(QPainter.SmoothPixmapTransform)

        self.setDragMode(QGraphicsView.RubberBandDrag)
        self.contents = {}
        self.paused = False
        self.queued_elements = []


    def replace_image(self, item_id, new_content):

        if item_id not in self.items:
            print("Unknown ID:", item_id)
            return
        
        

        item = self.contents[item_id]

        item.update_pixmap(new_content)
    
    def load_images(self,data):
        """
        Preload images into canvas
        """
        if self.paused:
            self.queued_elements.append(data)
            return

        id,content, x, y = data
        pixmap = cv2_to_qpixmap(content)

        if id not in self.contents: 
            scene_rect = self.scene.sceneRect()
            item = ImageItem(id,pixmap, scene_rect)
            item.setPos(x, y)
            self.scene.addItem(item)
            self.contents[id] = item

        else: 
            item = self.contents[id]
            item.update_pixmap(pixmap)

    def remove_image(self, item_id:int):
        """
        Remove an image item by ID
        """
        if self.paused:
            self.queued_elements.append(item_id)
            return

        if item_id not in self.contents:
            print(f"[remove_image] ID not found: {item_id}")
            return False

        item = self.contents[item_id]
        self.scene.removeItem(item)
        item.setParentItem(None)
        del item
        del self.contents[item_id]

        return True
    
    def pause(self):
        self.paused = not self.paused
        if not self.paused:
            for item in self.queued_elements:
                if isinstance(item,int):
                    self.remove_image(item)
                else:
                    self.load_images(item)
            self.queued_elements = []



class MainWindow(QMainWindow):

    def __init__(self,dimension_x,dimension_y):
        super().__init__()

        self.setWindowTitle("Board")

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout()
        central_widget.setLayout(main_layout)

        self.view = CanvasView(dimension_x - 50,dimension_y - 50)
        self.view.setFixedSize(dimension_x ,dimension_y)
        main_layout.addWidget(self.view)

        button_panel = QWidget()
        button_layout = QVBoxLayout()
        button_panel.setLayout(button_layout)
        pause_button = QPushButton("P")
        btn2 = QPushButton("A")
        pause_button.clicked.connect(self.pause)
        btn2.clicked.connect(self.screenshot)
        button_layout.addWidget(pause_button)
        button_layout.addWidget(btn2)
        button_layout.addStretch()

        main_layout.addWidget(button_panel) 

        self.resize(dimension_x + 100, dimension_y)

    def pause(self):
        self.view.pause()
        if self.view.paused:
            self.pause_button.setText("R")
        else:
            self.pause_button.setText("P")


    def screenshot(self):
        pass

    



from PySide6.QtCore import QObject, Signal, Slot, QThread
import cv2
from pipeline.orchestrator import run_orchestrator



class Worker(QObject):
    finished = Signal()
    load_content = Signal(tuple)
    remove_content = Signal(tuple)
    corners = None
    def __init__(self,corners):
        super().__init__()
        self.corners = corners

    @Slot()
    def run(self):
        # run_orchestrator(self.corners,self)
        # print("Worker Finished")
        # self.finished.emit()
        image1 = cv2.imread("input.jpg")
        image2 = cv2.imread("image2.png")
        image3 = cv2.imread("image3.png")
        data1 = (1,image1,100,100)
        data2 = (2,image2,500,300)
        data3 = (2,image3,500,300)
        self.load_content.emit(data1)
        self.load_content.emit(data2)
        time.sleep(5)
        self.remove_content.emit(1)
        self.load_content.emit(data3)
        # for i in range(20):
        #     time.sleep(1)
        #     if not i%3:
        #         self.load_content.emit(data2)
        #     elif not (i-1)%3:
        #         self.load_content.emit(data1)
        #     else:
        #         self.load_content.emit(data3)

import numpy as np
from preprocessing.corner_selection import select_board_corners

def run_gui(video_path: str = "pipeline/Test3.mp4",enable_gui = True):

    cap = cv2.VideoCapture(video_path)
    ok, first_frame = cap.read()
    if not ok:
        raise FileNotFoundError(f"Could not read video: {video_path}")
    
    print("[INFO] Select board corners...")
    corners = select_board_corners(first_frame)
    if corners is None:
        print("[INFO] Corner selection aborted.")
        return

    tl, tr, br, bl = corners
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    x = int(max(widthA, widthB))

    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    y = int(max(heightA, heightB))

    app = QApplication(sys.argv)
    win = MainWindow(x, y)

    thread = QThread()
    worker = Worker(corners)

    worker.moveToThread(thread)

    thread.started.connect(worker.run)
    worker.load_content.connect(win.view.load_images)
    worker.remove_content.connect(win.view.remove_image)
    worker.finished.connect(thread.quit)


    thread.start()
    if enable_gui:
        win.show()
        sys.exit(app.exec())
    else:
        print("Waiting")
        thread.finished.connect(app.quit)
        app.exec()   # run the event loop until app.quit() is called
        print("Quitting")
        sys.exit()
    
    




        
