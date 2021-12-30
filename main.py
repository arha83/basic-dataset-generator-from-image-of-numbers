import cv2 as cv
import numpy as np

def stackImages(columns, rows, *images):
    imageIndex= 0
    horizontals= []
    for j in range(columns):
        blocks=[]
        for i in range(rows):
            blocks.append(images[imageIndex])
            imageIndex += 1
        horizontals.append(np.hstack(blocks))
    canvas= np.vstack(horizontals)
    return canvas

    

baseImage= cv.imread('./numbers.png')
baseImage= cv.cvtColor(baseImage, cv.COLOR_BGR2GRAY)
invert= cv.bitwise_not(baseImage)
blur= cv.GaussianBlur(baseImage, (15,15), 0)
_, thresh= cv.threshold(blur, 230, 255, cv.THRESH_BINARY)

win= stackImages(2, 2, baseImage, invert, blur, thresh)
cv.imshow('images', win)
cv.waitKey(0)