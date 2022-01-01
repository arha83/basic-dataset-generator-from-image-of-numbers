import cv2 as cv
import numpy as np

# sorting rectangles of numbers
def sortRectangles(rectangles_shuffled):
    rectangles_xSorted= sorted(rectangles_shuffled, key= lambda x: x[0]) #sorting in x axis.
    rectangles_ySorted= []
    for i in range(10):
        rectangles_y= [rectangles_xSorted[i*5+j] for j in range(len(rectangles_shuffled)//10)]
        rectangles_ySorted.append(sorted(rectangles_y, key= lambda x: x[1])) # sorting in y axis.
    rectangles=[]
    for i in range(len(rectangles_shuffled)//10): # rearranging rectangles 
        rectanglesRow= [rectangles_ySorted[j][i] for j in range(10)]
        rectangles+= rectanglesRow
    return rectangles
    

# images to work on
baseImage= cv.imread('./numbers.png')
baseImage= cv.copyMakeBorder(baseImage, 20,20,20,20, cv.BORDER_CONSTANT, value=(255,255,255))
baseImage_color= np.copy(baseImage)
baseImage= cv.cvtColor(baseImage, cv.COLOR_BGR2GRAY)
invert= cv.bitwise_not(baseImage)
# thresholding blured images for generalizing and 
# removing noise
blur= cv.GaussianBlur(baseImage, (7,7), 0)
_, thresh= cv.threshold(blur, 200, 255, cv.THRESH_BINARY)
# finding all contours with all hierarchies
contours_all, hierarchies= cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
# filtering numbers' contours
contours= []
for index, contour in enumerate(contours_all):
    hierarchy= hierarchies[0][index] # ( Next , Previous , Next_Child , Parent )
    hasGrandParents= hierarchies[0][hierarchy[3]][0] != -1
    if hierarchy[3] != -1 and (hierarchy[0] != -1 or hierarchy[1] != -1) and not hasGrandParents:
        contours.append(contour)
# getting bounding rectangles of numbers
rectangles_shuffled= []
for contour in contours:
    x, y, w, h= cv.boundingRect(contour)
    rectangles_shuffled.append((x, y, w, h))
    cv.rectangle(baseImage_color, (x,y), (x+w,y+h), (255,0,0), 1)
# sorting rectangles from top-left to bottom-right
rectangles= sortRectangles(rectangles_shuffled)
# spliting inverted image to 32*32 images
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
# saving datasets
np.save('./dataset.npy', images)
np.save('./labels.npy', labels)
# presenting results
for index, rectangle in enumerate(rectangles):
    cv.putText(baseImage_color, str(index), (rectangle[0],rectangle[1]), cv.FONT_HERSHEY_SIMPLEX, 0.35, (255,0,0))
cv.drawContours(baseImage_color, contours_all, -1, (0,255,0), 1)
cv.drawContours(baseImage_color, contours, -1, (0,0,255), 1)
cv.imshow('images', baseImage_color)
cv.waitKey(0)