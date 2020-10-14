# SOURCES:
# https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_calib3d/py_calibration/py_calibration.html
# https://medium.com/@kennethjiang/calibrate-fisheye-lens-using-opencv-333b05afa0b0


###################################################################################################################
# PARAMETERS
###################################################################################################################

CHECKERBOARD = (6,9)  # NUMBER OF VERTICES; NOT SQUARES!!!
SHOW_IMAGES = False

###################################################################################################################


import numpy as np
import cv2
import glob


# SETUP
print( "Starting calibration setup..." )

subpix_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)
calib_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1.0e-6)
calibration_flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC+cv2.fisheye.CALIB_CHECK_COND+cv2.fisheye.CALIB_FIX_SKEW

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((1,CHECKERBOARD[0]*CHECKERBOARD[1],3), np.float32)
objp[0,:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane

images = glob.glob('./calibrationImages/*.jpeg')

for fname in images:
  print( fname )
  img = cv2.imread( fname )
  gray = cv2.cvtColor( img , cv2.COLOR_BGR2GRAY )
  
  # Find the chess board corners
  ret, corners = cv2.findChessboardCorners( gray , CHECKERBOARD , None )
  
  # If found, add object points, image points (after refining them)
  if ret == True:
    objpoints.append(objp)
    
    corners2 = cv2.cornerSubPix( gray , corners , (11,11) , (-1,-1) , subpix_criteria )
    imgpoints.append(corners2)
    
    # Draw and display the corner<s
    if SHOW_IMAGES:
      img = cv2.drawChessboardCorners( img , CHECKERBOARD , corners2 , ret )
      cv2.imshow( fname , img )
    #key = cv2.waitKey(1)
    key = cv2.waitKey()
    if key == ord('p'):
      key = cv2.waitKey(-1)  # wait until any key is pressed
    if key == ord('q'):
      break

cv2.destroyAllWindows()
print( "Finished calibration setup." )

# CALIBRATION
print( "Starting calibration..." )
K = np.zeros((3, 3))
D = np.zeros((4, 1))
N_OK = len(objpoints)
rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]
tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]
rms, _, _, _, _ = cv2.fisheye.calibrate( objpoints, imgpoints, gray.shape[::-1], K, D, rvecs, tvecs, calibration_flags, calib_criteria )
print( "Finished calibration." )

import pickle
with open( 'calibration.pkl' , 'wb' ) as f:
  pickle.dump( [ K , D ] , f )
