
###################################################################################################################
# PARAMETERS
###################################################################################################################

VIDEO_SIZE = (960,540)
GENERATE_VIDEOS = True
DETECT_CORNERS = False
VIDEOS_PATH = "./generatedVideos/"

###################################################################################################################


import numpy as np
import cv2
import pickle


# we load the calibration
with open( 'calibration.pkl' , 'rb' ) as f:
  K, D = pickle.load( f )

# we capture the video
vidcap = cv2.VideoCapture( 'calibrationVideo.MP4' )
success, img = vidcap.read()

# we prepare things if we want to generate a video
if GENERATE_VIDEOS == True:
  import os
  if not os.path.exists( VIDEOS_PATH ):
    os.makedirs( VIDEOS_PATH )
  fps = vidcap.get( cv2.CAP_PROP_FPS )
  vid = cv2.VideoWriter( VIDEOS_PATH + '/undistorted.mp4' , cv2.VideoWriter_fourcc(*'DIVX') , fps , VIDEO_SIZE )

# we run over the video
while success:
  # we undistort the image
  h, w = img.shape[:2]
  new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(K, D, (w, h), np.eye(3), balance=1, new_size=VIDEO_SIZE, fov_scale=1.04 )
  map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), new_K, VIDEO_SIZE, cv2.CV_16SC2)
  udst = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
  
  # we detect the corners
  if DETECT_CORNERS:
    gray = cv2.cvtColor(udst,cv2.COLOR_BGR2GRAY)
    dst = cv2.cornerHarris(gray,2,3,0.04)
    #result is dilated for marking the corners, not important
    dst = cv2.dilate(dst,None)
    # Threshold for an optimal value, it may vary depending on the image.
    udst[dst>0.01*dst.max()]=[0,0,255]
  
  # we draw
  cv2.imshow( 'undistorted' , udst )
  vid.write( udst )
  
  # we check if want to play/pause, or quit the video
  key = cv2.waitKey(1)
  if key == ord('p'):
    key = cv2.waitKey(-1)  # wait until any key is pressed
  if key == ord('q'):
    break
  
  # we take the next image in the video
  success, img = vidcap.read()

vidcap.release()
vid.release()
cv2.destroyAllWindows()
