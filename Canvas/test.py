import io
import matplotlib.pyplot as plt

from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import QGraphicsPixmapItem


class LatexGraphicsItem(QGraphicsPixmapItem):
    def __init__(self, latex, parent=None):
        pixmap = self.render_latex(latex)
        super().__init__(pixmap, parent)

        # make it behave like QGraphicsTextItem
        self.setFlags(
            QGraphicsPixmapItem.ItemIsMovable |
            QGraphicsPixmapItem.ItemIsSelectable 
        )

    def render_latex(self, latex):
        fig = plt.figure()
        fig.text(0, 0, f"${latex}$")

        buffer = io.BytesIO()
        fig.savefig(buffer, format="png", bbox_inches="tight", pad_inches=0)
        plt.close(fig)

        pixmap = QPixmap()
        pixmap.loadFromData(buffer.getvalue())
        return pixmap

from PySide6.QtWidgets import QApplication, QGraphicsScene, QGraphicsView

app = QApplication([])

scene = QGraphicsScene()

latex_item = LatexGraphicsItem(r"\frac{a^2 + b^2}{c}")
scene.addItem(latex_item)

view = QGraphicsView(scene)
view.show()

app.exec()