import cv2
import os
import sys
import imutils
from scipy.spatial import distance as dist
import numpy as np
#     column_name = ['nose', 'left shoulder', 'right shoulder', 'left elbow', 'right elbow', 'left wrist', 'right wrist',
#                    'left knee', 'right knee', 'left ankle', 'right ankle']
def braco_relaxado(image, mask_output, x, y):
    right_shoulder_x = x[2]
    right_shoulder_y = y[2]
    right_elbow_x = x[4]
    right_elbow_y = y[4]

    braco_relax_right_x = int((right_shoulder_x + right_elbow_x) / 2)
    braco_relax_right_y = int((right_shoulder_y + right_elbow_y) / 2)

    output = cv2.cvtColor(mask_output, cv2.COLOR_BGR2GRAY)

    # Blue color in BGR
    color = (0, 0, 255)
    # Line thickness of 2 px
    thickness = 2
    start_point = (braco_relax_right_x - 10, braco_relax_right_y)
    end_point = (braco_relax_right_x, braco_relax_right_y + 10)

    # Draw a rectangle with blue line borders of thickness of 2 px
    mask = cv2.rectangle(image, start_point, end_point, color, thickness)

    cropped_output = output[start_point[1]:end_point[1], start_point[0]:end_point[0]]

    cnts = cv2.findContours(cropped_output, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    c = max(cnts, key=cv2.contourArea)
    x_cont_pantu = c[:, :, 0]
    half_contour = int(len(c) / 2)
    extLeft = tuple(c[x_cont_pantu[0:half_contour].argmax()][0])

    p1 = int(start_point[0] + extLeft[0])
    p2 = int(start_point[1] + extLeft[1])

    start_point2 = (braco_relax_right_x, braco_relax_right_y)
    end_point2 = (braco_relax_right_x+10, braco_relax_right_y)

    p3 = end_point2[0]
    p4 = end_point2[1]
    bra_relax1 = (p1, p2)
    bra_relax2 = (p3, p4)
    cv2.circle(image, (p1, p2), 3, (255, 255, 0), -1)
    cv2.circle(image, (p3, p4), 3, (255, 255, 0), -1)
    cv2.imshow('mask', image)
    cv2.waitKey(0)
    return dist.euclidean(bra_relax1, bra_relax2)