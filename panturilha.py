import cv2
import os
import sys
import imutils
from scipy.spatial import distance as dist
import numpy as np

# column_name = ['nose', 'left shoulder', 'right shoulder', 'left hip', 'right hip', 'left knee', 'right knee',
#                    'left ankle', 'right ankle']

def pantu_max(image, mask_output, x, y):
    right_knee_x = x[6]
    right_knee_y = y[6]
    right_ankle_x = x[8]
    right_ankle_y = y[8]

    output = cv2.cvtColor(mask_output, cv2.COLOR_BGR2GRAY)

    # Blue color in BGR
    color = (0, 0, 255)
    # Line thickness of 2 px
    thickness = 2
    start_point = (right_knee_x-30, right_knee_y)
    end_point = (right_ankle_x, right_ankle_y)

    # Draw a rectangle with blue line borders of thickness of 2 px
    mask = cv2.rectangle(image, start_point, end_point, color, thickness)

    cropped_output = output[start_point[1]:end_point[1], start_point[0]:end_point[0]]

    cnts = cv2.findContours(cropped_output, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    c = max(cnts, key=cv2.contourArea)
    x_cont_pantu = c[:, :, 0]
    half_contour = int(len(c)/2)
    extLeft = tuple(c[x_cont_pantu[0:half_contour].argmin()][0])

    p1 = int(start_point[0]+extLeft[0])
    p2 = int(start_point[1]+extLeft[1])

    start_point2 = (right_knee_x, p2)
    end_point2 = (right_ankle_x+30, right_ankle_y)

    # Draw a rectangle with blue line borders of thickness of 2 px
    mask = cv2.rectangle(image, start_point2, end_point2, color, thickness)

    cropped_output2 = output[start_point2[1]:end_point2[1], start_point2[0]:end_point2[0]]

    cnts2 = cv2.findContours(cropped_output2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts2 = imutils.grab_contours(cnts2)
    c2 = max(cnts2, key=cv2.contourArea)
    x_cont_pantu2 = c2[:, :, 0]
    half_contour2 = int(len(c2)/2)
    extLeft2 = tuple(c[x_cont_pantu2[half_contour2+1:len(c2)-1].argmin()][0])

    p3 = int(start_point2[0]+extLeft2[0])
    p4 = int(start_point2[1]+extLeft2[1])

    pantu_max_1 = [p1, p2]
    pantu_max_2 = [p3, p4]

    return dist.euclidean(pantu_max_1, pantu_max_2)