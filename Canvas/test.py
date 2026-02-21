from canvas import run, load_image

import cv2
import time

run()
time.sleep(0.5)
image1 = cv2.imread("input.jpg")
image2 = cv2.imread("image2.png")
image3 = cv2.imread("image3.png")
data1 = (1,image1,100,100)
data2 = (2,image2,500,300)
data3 = (2,image3,500,300)

load_image(data1)
load_image(data2)

for i in range(10):
    if i%2:
        load_image(data3)
    else:
        load_image(data2)
