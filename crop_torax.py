import cv2
import os
import sys
import imutils
from scipy.spatial import distance as dist
import numpy as np

def cint_minima(image,mask_output, x_lmark, y_lmark, image_width, image_height):
    start_row = int(x_lmark[1] * image_width)
    end_row = int(x_lmark[4] * image_width)
    star_column = int(y_lmark[1] * image_height)
    end_column = int(y_lmark[4] * image_height)

    right_shoulder_x = int(x_lmark[2] * image_width)
    right_shoulder_y = int(y_lmark[2] * image_height)

    output = cv2.cvtColor(mask_output, cv2.COLOR_BGR2GRAY)

    D1 = abs(end_column - star_column)
    D2 = abs(right_shoulder_x - end_row)

    # Blue color in BGR
    color = (0, 0, 255)
    # Line thickness of 2 px
    thickness = 2
    start_point = (right_shoulder_x-10, star_column+35)
    end_point = (right_shoulder_x + D2, star_column + D1-15)

    # Draw a rectangle with blue line borders of thickness of 2 px
    mask = cv2.rectangle(image, start_point, end_point, color, thickness)
    cropped_output = output[start_point[1]:end_point[1], start_point[0]:end_point[0]]

    # cv2.imwrite("cropped.png", image_output)

    cnts = cv2.findContours(cropped_output, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    c = max(cnts, key=cv2.contourArea)
    x_cont_coxa = c[:, :, 0]
    half_contour = int(len(c)/2)
    extLeft = tuple(c[x_cont_coxa[0:half_contour].argmax()][0])

    p1 = int(start_point[0]+extLeft[0])
    p2 = int(start_point[1]+extLeft[1])

    # start_point2 = (start_row-10, right_shoulder_y+35)
    start_point2 = (start_row-20, p2-10)
    end_point2 = (start_row + D2, star_column + D1-15)

    # Draw a rectangle with blue line borders of thickness of 2 px
    mask2 = cv2.rectangle(image, start_point2, end_point2, color, thickness)
    # cv2.imshow('mask', image)
    # cv2.waitKey(0)

    cropped_output2 = output[start_point2[1]:end_point2[1], start_point2[0]:end_point2[0]]
    image_output = image[start_point[1]:end_point2[1], start_point[0]:end_point2[0]]

    cnts2 = cv2.findContours(cropped_output2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts2 = imutils.grab_contours(cnts2)
    c2 = max(cnts2, key=cv2.contourArea)
    x_cont_coxa2 = c2[:, :, 0]
    half_contour2 = int(len(c2)/4)
    extLeft2 = tuple(c2[x_cont_coxa2[half_contour2:len(c2)-1].argmax()][0])

    p3 = int(start_point2[0]+extLeft2[0])
    p4 = int(start_point2[1]+extLeft2[1])

    cint_min_1 = [p1, p2]
    cint_min_2 = [p3, p4]
    # cv2.circle(image, (p1, p2), 3, (0, 0, 255), -1)
    # cv2.circle(image, (p3, p4), 3, (255, 0, 255), -1)
    # cv2.imshow('mask', image)
    # cv2.waitKey(0)
    return dist.euclidean(cint_min_1, cint_min_2)



