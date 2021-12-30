import cv2 as cv
import numpy as np

    

baseImage= cv.imread('./numbers.png')
baseImage= cv.copyMakeBorder(baseImage, 20,20,20,20, cv.BORDER_CONSTANT, value=(255,255,255))
baseImage_color= np.copy(baseImage)
baseImage= cv.cvtColor(baseImage, cv.COLOR_BGR2GRAY)
invert= cv.bitwise_not(baseImage)
blur= cv.GaussianBlur(baseImage, (7,7), 0)
_, thresh= cv.threshold(blur, 200, 255, cv.THRESH_BINARY)
contours_all, hierarchies= cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
contours= []
for index, contour in enumerate(contours_all):
    hierarchy= hierarchies[0][index]
    if hierarchy[3] != -1 and (hierarchy[0] != -1 or hierarchy[1] != -1):
        contours.append(contour)


cv.drawContours(baseImage_color, contours_all, -1, (0,255,0), 1)
cv.imshow('images', baseImage_color)
cv.waitKey(0)