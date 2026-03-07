import sys
import subprocess
from PySide6.QtCore import QPointF
from PySide6.QtWidgets import (
    QApplication,
    QGraphicsItem,
    QGraphicsView,
    QGraphicsScene,
    QGraphicsTextItem,
    QGraphicsPixmapItem,
    QMainWindow,
    QWidget,
    QPushButton,
    QLabel,
    QVBoxLayout,
    QHBoxLayout,
    QGraphicsTextItem,
    QLineEdit
)
import cv2
import numpy as np
from PySide6.QtGui import QPainter
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtCore import Qt, QTimer
import matplotlib.pyplot as plt

import time

import matplotlib.mathtext as mathtext
import io

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


parser = mathtext.MathTextParser("path")
def is_valid_latex(expr):
    try:
        parser.parse(expr)
        return True
    except Exception:
        return False

class ImageItem(QGraphicsPixmapItem):

    def __init__(self, item_id, image, scene):
        pixmap = cv2_to_qpixmap(image)
        super().__init__(pixmap)

        self.scene_ref = scene
        self.item_id = item_id

        self.setFlags(
            QGraphicsPixmapItem.ItemIsMovable |
            QGraphicsPixmapItem.ItemIsSelectable |
            QGraphicsPixmapItem.ItemSendsGeometryChanges
        )

    
    def update_content(self, new_pixmap):

        old_pos = self.pos()
        self.setPixmap(new_pixmap)
        self.setPos(old_pos)

class TextItem(QGraphicsTextItem):
    def __init__(self, item_id, text,size = 16) -> None:
        super().__init__(text)
        self.item_id = item_id

        self.setFlags(
            QGraphicsPixmapItem.ItemIsMovable |
            QGraphicsPixmapItem.ItemIsSelectable 
        )
        font = self.font()
        font.setPointSize(size)
        self.setFont(font)

    def update_content(self, new_text):

        old_pos = self.pos()
        self.setPlainText(new_text)
        self.setPos(old_pos)

class LatexItem(QGraphicsPixmapItem):
    def __init__(self, item_id, latex, size = 1, parent=None):
        pixmap = self.render_latex(latex,size)
        super().__init__(pixmap, parent)
        self.item_id = item_id
        self.latex_value = latex

        # make it behave like QGraphicsTextItem
        self.setFlags(
            QGraphicsPixmapItem.ItemIsMovable |
            QGraphicsPixmapItem.ItemIsSelectable 
        )

    def render_latex(self, latex,size = 16):
        fig = plt.figure(dpi=size*40)
        fig.text(0, 0, f"${latex}$")

        buffer = io.BytesIO()
        fig.savefig(buffer, format="png", bbox_inches="tight", pad_inches=0)
        plt.close(fig)

        pixmap = QPixmap()
        pixmap.loadFromData(buffer.getvalue())
        return pixmap

    def update_content(self, latex):
        old_pos = self.pos()
        new_pixmap = self.render_latex(latex)
        self.setPixmap(new_pixmap)
        self.setPos(old_pos)

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
        self.removed_elements_id = []


    def replace_image(self, item_id, new_content):

        if item_id not in self.items:
            print("Unknown ID:", item_id)
            return
        
        item = self.contents[item_id]

        item.update_content(new_content)
    
    def load_images(self,data:dict):
        """
        Preload images into canvas
        """
        if data["id"] in self.removed_elements_id:
            return
        if self.paused:
            self.queued_elements.append(data)
            return

        
        if data["id"] not in self.contents: 
            if data["type"] == "image":
                scene_rect = self.scene.sceneRect()
                item = ImageItem(data["id"],data["content"], scene_rect)
                item.setPos(data["x"], data["y"])
                self.scene.addItem(item)
                self.contents[data["id"]] = item
            elif data["type"] == "text":
                item = TextItem(data["id"],data["content"],data["size"] if data["size"] else 16)
                item.setPos(data["x"], data["y"])
                self.scene.addItem(item)
                self.contents[data["id"]] = item
            elif data["type"] == "latex":
                item = LatexItem(data["id"],data["content"],data["size"] if data["size"] else 16)
                item.setPos(data["x"], data["y"])
                self.scene.addItem(item)
                self.contents[data["id"]] = item

        else: 
            item = self.contents[data["id"]]
            item.update_content(data["content"])

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
        self.removed_elements_id.append(item_id)

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

    def remove_selected(self):
        selected_items = self.scene.selectedItems()
        for item in selected_items:
            if item.item_id >= 0:
                self.remove_image(item.item_id)
            else:
                self.scene.removeItem(item)

    def add_elements(self,element):
        if isinstance(element,str):
            text_item = QGraphicsTextItem(element)
            text_item.setFlag(QGraphicsTextItem.ItemIsSelectable, True)

            text_item.setFlag(QGraphicsTextItem.ItemIsMovable, True)
            text_item.setPos(50,50)

            font = text_item.font()
            font.setPointSize(20)
            text_item.setFont(font)
            text_item.item_id = -1
            self.scene.addItem(text_item)
        

class MainWindow(QMainWindow):

    def __init__(self,dimension_x,dimension_y):
        super().__init__()

        self.setWindowTitle("Board")

        #Main interface
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout()
        central_widget.setLayout(main_layout)

        self.view = CanvasView(dimension_x - 50,dimension_y - 50)
        self.view.setFixedSize(dimension_x ,dimension_y)
        main_layout.addWidget(self.view)

        #Buttons 
        button_panel = QWidget()
        button_layout = QVBoxLayout()
        button_panel.setLayout(button_layout)

        self.pause_button = QPushButton("Pause")
        self.pause_button.clicked.connect(self.pause)
        button_layout.addWidget(self.pause_button)

        remove = QPushButton("Remove Selected")
        remove.clicked.connect(self.remove_selected)
        button_layout.addWidget(remove)

        self.add_word_textbox = QLineEdit()
        button_layout.addWidget(self.add_word_textbox)

        add_text = QPushButton("Add Text")
        add_text.clicked.connect(self.add_text)
        button_layout.addWidget(add_text)
        button_layout.addStretch()

        main_layout.addWidget(button_panel) 

        self.resize(dimension_x + 100, dimension_y)

    def pause(self):
        self.view.pause()
        if self.view.paused:
            self.pause_button.setText("Resume")
        else:
            self.pause_button.setText("Pause")

    def remove_selected(self):
        self.view.remove_selected()

    def add_text(self):
        text = self.add_word_textbox.text()
        self.view.add_elements(text)

    
if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MainWindow(1200, 800)
    image1 = cv2.imread("input.jpg")
    image2 = cv2.imread("image2.png")
    image3 = cv2.imread("image3.png")
    data1 = {"id":1,"type":"image","content":image1,"x":200,"y":300}
    data2 = {"id":2,"type":"image","content":image2,"x":200,"y":600}
    data3 = {"id":3,"type":"image","content":image3,"x":800,"y":300}
    data4 = {"id":4,"type":"text","content":"Hello World","size":16,"x":800,"y":500}
    data5 = {"id":5,"type":"latex","content":r"\frac{a}{b}","size":10,"x":800,"y":500}
    
    win.view.load_images(data1)
    win.view.load_images(data2)
    win.view.load_images(data3)
    win.view.load_images(data4)
    win.view.load_images(data5)
    win.show()
    sys.exit(app.exec())   


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
        data1 = {"id":3,"content":image1,"x":200,"y":300}
        data2 = {"id":2,"content":image2,"x":200,"y":300}
        data3 = {"id":1,"content":image3,"x":800,"y":300}
        self.load_content.emit(data3)
        self.load_content.emit(data2)
        # time.sleep(5)
        # self.remove_content.emit(1)
        # self.load_content.emit(data3)
        for i in range(20):
            time.sleep(1)
            if not i%2:
                self.load_content.emit(data2)
            else:
                self.load_content.emit(data1)


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
        app.exec()   
        print("Quitting")
        sys.exit()
    




        
