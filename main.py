import cv2 as cv
import numpy as np


def sortRectangles(rectangles_shuffled):
    rectangles_xSorted= sorted(rectangles_shuffled, key= lambda x: x[0])
    rectangles_ySorted= []
    for i in range(10):
        rectangles_y= [rectangles_xSorted[i*5+j] for j in range(len(rectangles_shuffled)//10)]
        rectangles_ySorted.append(sorted(rectangles_y, key= lambda x: x[1]))
    rectangles=[]
    for i in range(len(rectangles_shuffled)//10):
        rectanglesRow= [rectangles_ySorted[j][i] for j in range(10)]
        rectangles+= rectanglesRow
    return rectangles
    


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

rectangles= sortRectangles(rectangles_shuffled)

imagesList= []
labelsList= []
for index, rectangle in enumerate(rectangles):
    (x, y, w, h) = rectangle
    num= invert[y:y+h,x:x+w]
    num= cv.copyMakeBorder(num, 10,10,10,10, cv.BORDER_CONSTANT, value=(0,0,0))
    num= cv.resize(num, (32,32))
    imagesList.append(num)
    labelsList.append(index % 10)
images= np.array(imagesList)
labels= np.array(labelsList)


for index, rectangle in enumerate(rectangles):
    cv.putText(baseImage_color, str(index), (rectangle[0],rectangle[1]), cv.FONT_HERSHEY_SIMPLEX, 0.35, (255,0,0))
cv.drawContours(baseImage_color, contours_all, -1, (0,255,0), 1)
cv.drawContours(baseImage_color, contours, -1, (0,0,255), 1)
cv.imshow('images', baseImage_color)
cv.waitKey(0)