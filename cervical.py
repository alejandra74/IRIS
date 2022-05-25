import cv2
import os
import sys
import imutils
from scipy.spatial import distance as dist
import numpy as np

def perimetro_cervical(image, mask_output, x, y):
    nose_x = x[0]
    nose_y = y[0]
    right_shoulder_x = x[2]
    right_shoulder_y = y[2]
    left_shoulder_x = x[1]
    left_shoulder_y = y[1]

    output = cv2.cvtColor(mask_output, cv2.COLOR_BGR2GRAY)

    # Blue color in BGR
    color = (0, 0, 255)
    # Line thickness of 2 px
    thickness = 2
    start_point = (right_shoulder_x, nose_y)
    end_point = (nose_x, left_shoulder_y)

    # Draw a rectangle with blue line borders of thickness of 2 px
    mask = cv2.rectangle(image, start_point, end_point, color, thickness)
    cropped_output = output[start_point[1]:end_point[1], start_point[0]:end_point[0]]

    cnts = cv2.findContours(cropped_output, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    c = max(cnts, key=cv2.contourArea)
    x_cont_coxa = c[:, :, 0]
    half_contour = int(len(c)/2)
    extLeft = tuple(c[x_cont_coxa[0:half_contour].argmax()][0])

    p1 = int(start_point[0]+extLeft[0])
    p2 = int(start_point[1]+extLeft[1])

    start_point2 = (nose_x, nose_y)
    end_point2 = (left_shoulder_x, left_shoulder_y)

    # Draw a rectangle with blue line borders of thickness of 2 px
    mask = cv2.rectangle(image, start_point2, end_point2, color, thickness)

    cropped_output2 = output[start_point2[1]:end_point2[1], start_point2[0]:end_point2[0]]

    cnts2 = cv2.findContours(cropped_output2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts2 = imutils.grab_contours(cnts2)
    c2 = max(cnts2, key=cv2.contourArea)
    x_cont_pantu2 = c2[:, :, 0]
    half_contour2 = int(len(c2) / 2)
    extLeft2 = tuple(c[x_cont_pantu2[half_contour2:len(c2) - 1].argmin()][0])

    p3 = int(start_point2[0] + extLeft2[0])
    p4 = int(start_point2[1] + extLeft2[1])

    cervical1 = [p1, p2]
    cervical2 = [p3, p4]

    # cv2.circle(image, (p1, p2), 3, (255, 255, 0), -1)
    # cv2.circle(image, (p3, p4), 3, (255, 255, 0), -1)
    # cv2.imshow('mask', image)
    # cv2.waitKey(0)
    return dist.euclidean(cervical1, cervical2)
