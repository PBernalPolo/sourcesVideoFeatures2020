import numpy as np
import cv2

img = cv2.imread('1403715596162142976.png')
newSize = ( int(1080*img.shape[1]/img.shape[0]) , 1080 )
img = cv2.resize( img , newSize )
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

threshold = 8000
edges = cv2.Canny( img , threshold*0.7 , threshold , apertureSize=7 )

cv2.imwrite( './edges.jpg' , edges )
cv2.imshow( "corners" , edges )
cv2.waitKey(0)
