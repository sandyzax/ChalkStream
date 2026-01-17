from enum import Enum
from typing import Any
from dataclasses import dataclass
import numpy as np
import cv2


class ContentType(Enum):
    WORDS = 1
    MATH = 2
    DIAGRAM = 3
    IMAGE = 4
@dataclass
class Content:
    id:int
    contentType:ContentType
    dimentions:tuple #(x,y,w,h)
    content:Any #change to the type of string, latex, and image 
    life_span:int

class ContentLoad:
    id_count:int
    contents:list[Content] = []
    board_size:tuple 
    bg_color:tuple #
    default_lifespan:int = 10

    def __init__(self,size:tuple,bg_color:tuple = (255,255,255)) -> None:
        self.id_count = 0
        self.board_size = size
        self.bg_color = bg_color


    
    def intersection_area(self, a, b)->float:
        ax, ay, aw, ah = a
        bx, by, bw, bh = b

        x1 = max(ax, bx)
        y1 = max(ay, by)
        x2 = min(ax + aw, bx + bw)
        y2 = min(ay + ah, by + bh)
        if x2 <= x1 or y2 <= y1:
            return 0
        return (x2 - x1) * (y2 - y1)

    def overlap_percentage(self, a, b)->float:
        inter = self.intersection_area(a, b)
        area_a = a[2] * a[3]
        area_b = b[2] * b[3]

        return inter / min(area_a, area_b)

    def update_frame(self,new_contents:list[Content])->None:
        updated_contents:list = []
        for new_content in new_contents:
            new = True
            for content in self.contents[:]:
                print(333)
                print(self.overlap_percentage(new_content.dimentions,content.dimentions))
                if new_content.contentType == content.contentType and self.overlap_percentage(new_content.dimentions,content.dimentions) > 0.6:
                    print(111)
                    content.dimentions = content.dimentions[:2] + new_content.dimentions[2:]
                    content.content = new_content.content
                    content.life_span = self.default_lifespan
                    updated_contents.append(content)
                    self.contents.remove(content)
                    new = False
                    break
            if new:
                updated_contents.append(Content(self.id_count,new_content.contentType,new_content.dimentions,new_content.content,self.default_lifespan))
                self.id_count = self.id_count + 1
        for content in self.contents:
            content.life_span -= 1
            if content.life_span > 0:
                updated_contents.append(content)
        print("xxx ",len(updated_contents))
        self.contents = updated_contents

    def create_content(self,content,content_type:ContentType,position:tuple)->Content:
        h, w = content.shape[:2]

        return Content(-1, content_type,(position[0],position[1],w,h),content,self.default_lifespan)

    def display_content(self):
        background = np.full((self.board_size[1], self.board_size[0], 3), self.bg_color, dtype=np.uint8)
        for content in self.contents:
            img = None
            if content.contentType == ContentType.IMAGE:
                img = content.content
            else:
                print(f"Content Type {content.contentType} Not Supported (yet), skipping")
                continue
            x,y,w,h = content.dimentions

            background[y:y+h, x:x+w] = img
        cv2.imshow("Result", background)
        cv2.waitKey(0)

def test1():
    overlay = cv2.imread("input.png")
    content_load = ContentLoad((1200,800))
    overlay = content_load.create_content(overlay,ContentType.IMAGE,(50,100))
    overlay_list = [overlay]
    content_load.update_frame(overlay_list)
    content_load.display_content()

def test2():
    img = cv2.imread("input.png")
    content_load = ContentLoad((1200,800))
    overlay = content_load.create_content(img,ContentType.IMAGE,(50,100))
    overlay_list = [overlay]
    content_load.update_frame(overlay_list)
    overlay = content_load.create_content(img,ContentType.IMAGE,(60,120))
    overlay_list = [overlay]
    content_load.update_frame(overlay_list)
    content_load.display_content()

def test3():
    img = cv2.imread("input.png")
    content_load = ContentLoad((1200,800))
    overlay = content_load.create_content(img,ContentType.IMAGE,(50,100))
    overlay_list = [overlay]
    content_load.update_frame(overlay_list)
    overlay = content_load.create_content(img,ContentType.IMAGE,(300,140))
    overlay_list = [overlay]
    content_load.update_frame(overlay_list)
    content_load.display_content()

if __name__ == "__main__":
    test1()
    # test2()
    # test3()

    

                    







