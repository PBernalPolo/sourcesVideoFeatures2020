# SETUP
#pip uninstall opencv-python
#pip uninstall opencv-contrib-python
#pip install opencv-python==3.4.2.17
#pip install opencv-contrib-python==3.4.2.17


###################################################################################################################
# PARAMETERS
###################################################################################################################
# the images should be downloaded first. In my case I downloaded from the following link:
# http://robotics.ethz.ch/~asl-datasets/ijrr_euroc_mav_dataset/vicon_room1/V1_02_medium/V1_02_medium.zip
IMAGES_PATH = "./V1_02_medium/mav0/cam0/data/"
GENERATE_VIDEOS = True
VIDEOS_PATH = "./generatedVideos/"
IMSHOW_ACTIVE = True
N_FEATURES = 200


DETECTOR = ""
#DETECTOR = "HARRIS"
#DETECTOR = "SHI-TOMASI"
#DETECTOR = "SIFT"
#DETECTOR = "SURF"
DETECTOR = "FAST"
#DETECTOR = "STAR"
#DETECTOR = "ORB"

DESCRIPTOR = ""
#DESCRIPTOR = "SIFT"
#DESCRIPTOR = "SURF"
DESCRIPTOR = "BRIEF"
#DESCRIPTOR = "ORB"

###################################################################################################################


import numpy as np
import cv2
import time
import glob


# we define the detector
if DETECTOR == "SIFT":
  det = cv2.xfeatures2d.SIFT_create( N_FEATURES )
elif DETECTOR == "SURF":
  det = cv2.xfeatures2d.SURF_create()
elif DETECTOR == "FAST":
  det = cv2.FastFeatureDetector_create()
elif DETECTOR == "STAR":
  det = cv2.xfeatures2d.StarDetector_create()
elif DETECTOR == "ORB":
  det = cv2.ORB_create()
else:
  det = cv2.AKAZE_create()
  #det = cv2.BRISK_create()
  #det = cv2.MSER_create()

# we define the descriptor
if DESCRIPTOR == "SIFT":
  desc = cv2.xfeatures2d.SIFT_create()
elif DESCRIPTOR == "SURF":
  desc = cv2.xfeatures2d.SURF_create()
elif DESCRIPTOR == "BRIEF":
  desc = cv2.xfeatures2d.BriefDescriptorExtractor_create()
elif DESCRIPTOR == "ORB":
  desc = cv2.ORB_create()
else:
  desc = cv2.AKAZE_create()
  #desc = cv2.BRISK_create()

# we create a Brute-Force Matcher
if  DESCRIPTOR == "ORB"  or  DESCRIPTOR == "BRIEF":
  matcher = cv2.BFMatcher( cv2.NORM_HAMMING , crossCheck=True )
else:
  matcher = cv2.BFMatcher( cv2.NORM_L2 , crossCheck=True )

# we take the name of the images
filenames = glob.glob( IMAGES_PATH + '*.png' )
filenames.sort( reverse=False )
# and we prepare things if we want to generate a video
if GENERATE_VIDEOS == True:
  import os
  if not os.path.exists( VIDEOS_PATH ):
    os.makedirs( VIDEOS_PATH )
  timestamps = np.array( [ int(s[len(IMAGES_PATH):s.rfind('.png')])  for s in filenames ] )
  frameRate = 1.0e9/(timestamps[1:]-timestamps[0:-1]).mean()
  img = cv2.imread( filenames[0] )
  vidDetector = cv2.VideoWriter( VIDEOS_PATH + '/' + DETECTOR+'+'+DESCRIPTOR+'_detection.mp4' , cv2.VideoWriter_fourcc(*'DIVX') , 20 , (img.shape[1],img.shape[0]) )
  vidMatchings = cv2.VideoWriter( VIDEOS_PATH + '/' + DETECTOR+'+'+DESCRIPTOR+'_matching.mp4' , cv2.VideoWriter_fourcc(*'DIVX') , 20 , (2*img.shape[1],img.shape[0]) )

# we run over the images
Nimages = 0
adtt = 0.0  # averaged detection time
adst = 0.0  # averaged descriptor time
amt = 0.0  # averaged matching time
att = 0.0  # averaged total time
auf = 0.0  # averaged update frequency
ailr = 0.0  # averaged inliers rate
kp0 = None
des0 = None
img0 = None
imgMatches = None
for name in filenames:
  # we obtain the image in gray
  img = cv2.imread( name , cv2.IMREAD_COLOR )
  gray = cv2.cvtColor( img , cv2.COLOR_BGR2GRAY )
  
  t0 = time.time()

  # we detect features
  if DETECTOR == "HARRIS":
    dst = cv2.cornerHarris( gray , 2 , 3 , 0.04 )
    ret, dst = cv2.threshold( dst , 0.01*dst.max() , 255 , 0 )
    dst = np.uint8(dst)
    # find centroids
    ret, labels, stats, centroids = cv2.connectedComponentsWithStats( dst )
    # define the criteria to stop and refine the corners
    criteria = ( cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER , 100 , 0.001 )
    corners = cv2.cornerSubPix( gray , np.float32(centroids) , (5,5) , (-1,-1) , criteria )
    kp = [ cv2.KeyPoint(x=c[0], y=c[1], _size=20) for c in corners[2:] ]  # the first one in corners is not a corner; it always lies on the center of the image (I don't know why)
  elif DETECTOR == "SHI-TOMASI":
    corners = cv2.goodFeaturesToTrack( gray , N_FEATURES , 0.01 , 10 )
    if type( corners ) == np.ndarray:
      kp = [ cv2.KeyPoint(x=c[0][0], y=c[0][1], _size=20) for c in corners ]
    else:
      kp = []
  else:
    kp = det.detect( gray , None )
  
  # we take the best N_FEATURES
  kp.sort( key=lambda x: x.response, reverse=True )
  kp = kp[0:N_FEATURES]
  
  t1 = time.time()
  
  # we obtain feature descriptors
  kp, des = desc.compute( gray , kp )

  t2 = time.time()
  
  # we match the features
  if len(kp) > 0:
    matches = matcher.match( des0 , des )
  
    t3 = time.time()
    
    # we measure some metrics for these matches
    if len(matches) >= 8.0:
      src_pts = np.float32([ kp0[m.queryIdx].pt for m in matches ]).reshape(-1,1,2)
      dst_pts = np.float32([ kp[m.trainIdx].pt for m in matches ]).reshape(-1,1,2)
      H, mask = cv2.findHomography( src_pts , dst_pts , cv2.RANSAC , 5.0 )
      diff = cv2.perspectiveTransform(src_pts,H)-dst_pts
      inliers = ( np.sqrt((diff*diff).sum(axis=2)) < img.shape[0]/10.0 )
      ilrate = inliers.sum()/len(inliers)
      #print( "Detection time (s):" , "{0:.5f}". format(t1-t0) , ";  descriptor time (s):" , "{0:.5f}". format(t2-t1) , ";  matching time (s):" , "{0:.5f}". format(t3-t2) , ";  total time (s):" , "{0:.5f}". format(t3-t0) , ";  update frequency (Hz):" , "{0:.3f}". format(1.0/(t3-t0)) , ";  inliers rate:" , "{0:.5f}". format(ilrate) )
      adtt += t1-t0
      adst += t2-t1
      amt += t3-t2
      att += t3-t0
      auf += 1.0/(t3-t0)
      ailr += ilrate
      Nimages += 1
    
  # we draw
  img = cv2.drawKeypoints( gray , kp , img , flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS )
  if IMSHOW_ACTIVE:
    cv2.imshow( DETECTOR , img )
  if len(kp) > 0:
    imgMatches = cv2.drawMatches( img0 , kp0 , img , kp , matches , imgMatches , matchColor=(0,255,0) , flags=2 )
    if IMSHOW_ACTIVE:
      cv2.imshow( "MATCHES" , imgMatches )
  if GENERATE_VIDEOS == True:
    vidDetector.write( img )
    vidMatchings.write( imgMatches )

  # we check if want to play/pause, or quit the video
  key = cv2.waitKey(1)
  if key == ord('p'):
    key = cv2.waitKey(-1)  # wait until any key is pressed
  if key == ord('q'):
    break
  
  # we update the previous values
  kp0 = kp
  des0 = des
  img0 = img
  

cv2.destroyAllWindows()
vidDetector.release()
vidMatchings.release()

print()
print( "Averaged detection time (s):" , "{0:.5f}". format(adtt/Nimages) , ";  averaged descriptor time (s):" , "{0:.5f}". format(adst/Nimages) , ";  averaged matching time (s):" , "{0:.5f}". format(amt/Nimages) , ";  averaged total time (s):" , "{0:.5f}". format(att/Nimages) , ";  averaged update frequency (Hz):" , "{0:.3f}". format(auf/Nimages) , ";  averaged inliers rate:" , "{0:.5f}". format(ailr/Nimages) )
