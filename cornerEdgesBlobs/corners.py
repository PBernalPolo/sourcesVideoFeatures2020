import numpy as np
import cv2

img = cv2.imread('1403715596162142976.png')
newSize = ( int(1080*img.shape[1]/img.shape[0]) , 1080 )
img = cv2.resize( img , newSize )
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

corners = cv2.goodFeaturesToTrack( gray , 100 , 0.01 , 10.0 )
corners = np.int0( corners )

for i in corners:
  x,y = i.ravel()
  cv2.circle( img , (x,y) , 6 , (0,0,255) , thickness=2 )

cv2.imwrite( './corners.jpg' , img )
cv2.imshow( "corners" , img )
cv2.waitKey(0)
