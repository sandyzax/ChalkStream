from PySide6.QtWidgets import QApplication, QGraphicsScene, QGraphicsView, QGraphicsPixmapItem
from PySide6.QtGui import QPixmap

app = QApplication([])
scene = QGraphicsScene()
view = QGraphicsView(scene)

# To make an image draggable, it's literally one line:
view.resize(1000, 600)
pixmap_item = QGraphicsPixmapItem(QPixmap("input.jpg"))
pixmap_item.setFlag(QGraphicsPixmapItem.GraphicsItemFlag.ItemIsMovable)
scene.addItem(pixmap_item)

view.show()
app.exec()