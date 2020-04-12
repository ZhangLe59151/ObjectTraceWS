# Demo2: Object tracking using correlation filter
import cv2
import numpy as np
from matplotlib import pyplot as plt
# %matplotlib inline
import time
import sys

def draw_flow(im,flow,step):
  h,w = im.shape
  y,x = np.mgrid[step:h:step,step:w:step].reshape(2,-1)
  fx,fy = flow[y,x].T
 
  # create line endpoints
  lines = np.vstack([x,y,x+fx,y+fy]).T.reshape(-1,2,2)
  lines = np.int32(lines)
 
  # create image and draw
  vis = cv2.cvtColor(im,cv2.COLOR_GRAY2BGR)
  for (x1,y1),(x2,y2) in lines:
    cv2.line(vis,(x1,y1),(x2,y2),(0,255,0),1)
    cv2.circle(vis,(x1,y1),1,(0,255,0), -1)
  return vis

# Load images
frame1 = cv2.imread('data/frame3.png',0)
frame2 = cv2.imread('data/frame4.png',0)

# Display the input images
plt.figure(figsize=(16,8))
plt.subplot(1, 2, 1)
plt.imshow(frame1,'gray')
plt.axis('off')
plt.title('frame1')
plt.subplot(1, 2, 2)
plt.imshow(frame2,'gray')
plt.axis('off')
plt.title('frame2')
plt.show()

# Use mouse to draw bounding box of the roi, press ESC to finish drawing
# Otherwise, you can manually specify the roi
#roi = [21, 211, 120, 70] # in [x, y, w, h] format
roi = cv2.selectROI(frame1, False)

methods = 'cv2.TM_CCORR'
obj_template = frame1[roi[1]:roi[1]+roi[3], roi[0]:roi[0]+roi[2]]
h, w = obj_template.shape

frame2_bk = frame2.copy()

result = cv2.matchTemplate(frame2_bk, obj_template, eval(methods))
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

# If the method is TM_SQDIFF, take minimum
if methods == 'cv2.TM_SQDIFF':
    top_left = min_loc
else:
    top_left = max_loc
bottom_right = (top_left[0] + w, top_left[1] + h)

cv2.rectangle(frame2_bk, top_left, bottom_right, 255, 5)

frame1_display = frame1.copy()
cv2.rectangle(frame1_display, (roi[0], roi[1]),(roi[0]+roi[2],roi[1]+roi[3]),255,5)

plt.figure(figsize=(16,8))
plt.subplot(1, 4, 1)
plt.imshow(obj_template,'gray')
plt.title('Target object template')
plt.axis('off')
plt.subplot(1, 4, 2)
plt.imshow(frame1_display,'gray')
plt.title('Target object in frame 1')
plt.axis('off')
plt.subplot(1,4,3)
plt.imshow(result, 'gray')
plt.title('Matching score map \n (' + methods + ')')
plt.axis('off')
plt.subplot(1,4,4)
plt.imshow(frame2_bk, 'gray')
plt.title('Detected object position in frame 2')
plt.axis('off')
plt.show()
cv2.destroyAllWindows()