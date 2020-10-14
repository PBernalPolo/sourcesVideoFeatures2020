import numpy as np
import cv2

img = cv2.imread( '1403715596162142976.png' )
newSize = ( int(1080*img.shape[1]/img.shape[0]) , 1080 )
img = cv2.resize( img , newSize )
gray = cv2.cvtColor( img , cv2.COLOR_BGR2GRAY )

detector = cv2.xfeatures2d.SIFT_create( 100 )
kpoints = detector.detect( gray , None )

Nkp = 0
for kp in kpoints:
  size = np.int( kp.size )
  if size > 3:
    x = np.int( kp.pt[0] )
    y = np.int( kp.pt[1] )
    cv2.circle( img , (x,y) , size , (0,0,255) , thickness=2 )
    Nkp += 1
print( Nkp )

cv2.imwrite( './blobs.jpg' , img )
cv2.imshow( "blobs" , img )
cv2.waitKey(0)
