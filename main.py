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

rectangles_shuffled= []
for contour in contours:
    x, y, w, h= cv.boundingRect(contour)
    rectangles_shuffled.append((x, y, w, h))
    cv.rectangle(baseImage_color, (x,y), (x+w,y+h), (255,0,0), 1)

for index, rectangle in enumerate(rectangles):
    cv.putText(baseImage_color, str(index), (rectangle[0],rectangle[1]), cv.FONT_HERSHEY_SIMPLEX, 0.25, (255,0,0))

print(rectangles, len(rectangles))
cv.drawContours(baseImage_color, contours_all, -1, (0,255,0), 1)
cv.drawContours(baseImage_color, contours, -1, (0,0,255), 1)
cv.imshow('images', baseImage_color)
cv.waitKey(0)