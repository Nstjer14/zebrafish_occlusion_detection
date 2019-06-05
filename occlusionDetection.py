# -*- coding: utf-8 -*-
"""
Created on Fri May 31 15:55:06 2019

@author: Niclas
"""

import numpy as np
import imutils
import cv2
from skimage.morphology import skeletonize_3d, skeletonize
#import sknw
from matplotlib import pyplot as plt
import csv

vidPath = '../videos/Narrow-3f-bg-50fps.avi'
contPath = 'contours/contours_3f_comb.csv'
intPath = 'intersections/intersections_3f_comb.csv'
vidObj = cv2.VideoCapture(vidPath)

occl_frame_nmbr = []
centre = []

def preProcess(image):
    img = image[167:972, 0:1920]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.threshold(blurred, 50, 255, cv2.THRESH_BINARY)[1]
    img = cv2.bitwise_not(thresh)
    binImg = img/255
    return img, binImg

def getContours(img):
    cnts = cv2.findContours(img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    return cnts

def findNeighbours(x,y,img):
    """Return 8-neighbours of image point P1(x,y), in a clockwise order"""
    n_img = img
    x_1, y_1, x1, y1 = x-1, y-1, x+1, y+1;
    return [ n_img[x_1][y], n_img[x_1][y1], n_img[x][y1], n_img[x1][y1], n_img[x1][y], n_img[x1][y_1], n_img[x][y_1], n_img[x_1][y_1] ]

def getSkeletonIntersection(skeleton):
    """ Given a skeletonised image, it will give the coordinates of the intersections of the skeleton.
    
    Keyword arguments:
    skeleton -- the skeletonised image to detect the intersections of
    
    Returns: 
    List of 2-tuples (x,y) containing the intersection coordinates
    """
    # A biiiiiig list of valid intersections             2 3 4
    # These are in the format shown to the right         1 C 5
    #                                                    8 7 6 
    validIntersection = [[0,1,0,1,0,0,1,0],[0,0,1,0,1,0,0,1],[1,0,0,1,0,1,0,0],
                         [0,1,0,0,1,0,1,0],[0,0,1,0,0,1,0,1],[1,0,0,1,0,0,1,0],
                         [0,1,0,0,1,0,0,1],[1,0,1,0,0,1,0,0],[0,1,0,0,0,1,0,1],
                         [0,1,0,1,0,0,0,1],[0,1,0,1,0,1,0,0],[0,0,0,1,0,1,0,1],
                         [1,0,1,0,0,0,1,0],[1,0,1,0,1,0,0,0],[0,0,1,0,1,0,1,0],
                         [1,0,0,0,1,0,1,0],[1,0,0,1,1,1,0,0],[0,0,1,0,0,1,1,1],
                         [1,1,0,0,1,0,0,1],[0,1,1,1,0,0,1,0],[1,0,1,1,0,0,1,0],
                         [1,0,1,0,0,1,1,0],[1,0,1,1,0,1,1,0],[0,1,1,0,1,0,1,1],
                         [1,1,0,1,1,0,1,0],[1,1,0,0,1,0,1,0],[0,1,1,0,1,0,1,0],
                         [0,0,1,0,1,0,1,1],[1,0,0,1,1,0,1,0],[1,0,1,0,1,1,0,1],
                         [1,0,1,0,1,1,0,0],[1,0,1,0,1,0,0,1],[0,1,0,0,1,0,1,1],
                         [0,1,1,0,1,0,0,1],[1,1,0,1,0,0,1,0],[0,1,0,1,1,0,1,0],
                         [0,0,1,0,1,1,0,1],[1,0,1,0,0,1,0,1],[1,0,0,1,0,1,1,0],
                         [1,0,1,1,0,1,0,0]];
    skeImg = skeleton
    #skeImg = skeImg/255;
    intersections = list();
    for x in range(1,len(skeImg)-1):
        for y in range(1,len(skeImg[x])-1):
            # If we have a white pixel
            if skeImg[x][y] == 1:
                neighbours = findNeighbours(x,y,skeImg);
                valid = True;
                if neighbours in validIntersection:
                    intersections.append((y,x));
    # Filter intersections to make sure we don't count them twice or ones that are very close together
    for point1 in intersections:
        for point2 in intersections:
            if (((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2) < 10**2) and (point1 != point2):
                intersections.remove(point2);
    # Remove duplicates
    intersections = list(set(intersections));
    return intersections;


def getContourCentre(img):
    
    cnts = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    
    for (cnt_i, cnt) in enumerate(cnts):
    	# draw the contour and center of the shape on the image
        cv2.drawContours(img, [cnt], -1, (0, 255, 0), 1)
        numb_cnts = cnt_i+1
        
    frame_nmbr = vidObj.get(cv2.CAP_PROP_POS_FRAMES)
    int(frame_nmbr)
    if numb_cnts < 3:
        cv2.rectangle(img,(0,0),(1920,805),(0,0,255),3)
        occl_frame_nmbr.append(frame_nmbr)
        largest = max(cnts, key=cv2.contourArea)
        #for (j, d) in enumerate(cnts):
        	# compute the center of the box
        bX,bY,bW,bH = cv2.boundingRect(largest)
        cX = int((bX+(bX+bW))/2)
        cY = int((bY+(bY+bH))/2)

    	# draw the bounding box and center of the box on the image
        cv2.circle(img, (cX, cY), 2, (255, 255, 255), -2)
        cv2.rectangle(img,(bX,bY),(bX+bW,bY+bH),(0,255,0),2)
        centre.append([cX,cY])
        
        
    return centre, occl_frame_nmbr

    

#Video reading loop - also main loop

vidObj = cv2.VideoCapture(vidPath)


count = 0
positions = []
countInts = 0
for i in range(3000): positions.append((0,0))

while True:
    ret, frame = vidObj.read()
    if frame is None:
        break
    #Using preProcess to extract greyscale and binary images
    img, binImg = preProcess(frame)
    
    
    #getting skeleton from the binary image
    skeleton = skeletonize(binImg)
    '''
    rows = len(skeleton[1])
    cols = len(skeleton)
    for i in range(cols):
        for j in range(rows):
            if skeleton[i][j] == 1:
                cv2.circle(img, (j,i), 0, (0, 255, 0), -2)
    '''
    #finding intersections from the skeleton. Looking 
    #for any valid intersection between two fish spines.
    intersections = getSkeletonIntersection(skeleton)
    for h in intersections:
        print(h[0],h[1])
    #append intersection coordinates 
    #together with frame number
    if len(intersections) > 0:
        countInts += 1
        for k in intersections:
            positions[count] = (k[0],k[1])
            
    
    if len(intersections) == 0:
        occlude_centre, occlude_frame = getContourCentre(img)
    
   
    
            
    print('Frame: ',count)
    count += 1
'''   

    cv2.imshow('video',img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
'''
 
vidObj.release()
cv2.destroyAllWindows() 
print('Finished')

posCount = 0
g = 0
with open(intPath, 'w', newline='') as csvfile:
    #fieldnames = ['x_position', 'y_position']
    writer = csv.writer(csvfile)
    #writer.writeheader()
    for position in positions:
        posCount += 1
        writer.writerow(position)
        #x, y = position[0], position[1]
        #writer.writerow({'x_position': x, 'y_position': y})
        #g += 1

with open(contPath, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    for bBoxCentre in occlude_centre:
        writer.writerow(bBoxCentre)

                   
print('finished writing data')
print(intPath)