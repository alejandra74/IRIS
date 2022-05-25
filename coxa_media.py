import cv2
import os
import sys
import imutils
from scipy.spatial import distance as dist
#     column_name = ['nose', 'left shoulder', 'right shoulder', 'left hip', 'right hip', 'left knee', 'right knee',
#                    'left ankle', 'right ankle']
def coxa_media(image, x_lmark, y_lmark, image_width, image_height, x_contour, y_contour, c):

    # image_width = 1280
    # image_height = 720
    x_left = []
    y_left = []
    x_right = []
    y_right = []

    # coordinates for braco relaxado esquerdo
    coord_left_x = [x_lmark[3], x_lmark[5]]
    coord_left_y = [y_lmark[3], y_lmark[5]]
    # coordinates for braco relaxado direito
    coord_right_x = [x_lmark[4], x_lmark[6]]
    coord_right_y = [y_lmark[4], y_lmark[6]]

    for i in range(len(coord_left_x)):
        x_left.append(int(coord_left_x[i] * image_width))
        y_left.append(int(coord_left_y[i] * image_height))
        x_right.append(int(coord_right_x[i] * image_width))
        y_right.append(int(coord_right_y[i] * image_height))

    coxa_left_x = int((x_left[0] + x_left[1]) / 2)
    coxa_left_y = int((y_left[0] + y_left[1]) / 2)

    coxa_right_x = int((x_right[0] + x_right[1]) / 2)
    coxa_right_y = int((y_right[0] + y_right[1]) / 2)

    x = [coxa_left_x, coxa_right_x]
    y = [coxa_left_y, coxa_right_y]

    # create empty arrays to save landmark extern points
    d = []
    e = []
    for i in range(int(len(c)/2)):
        for j in range(len(y)):
            if y_contour[i][0] == y[j]:
                d.append(x_contour[i][0])
                e.append(y_contour[i][0])
            if y_contour[i][0] + 1 == y[j]:
                d.append(x_contour[i][0])
                e.append(y_contour[i][0])
            if y_contour[i][0] - 1 == y[j]:
                d.append(x_contour[i][0])
                e.append(y_contour[i][0])

    # for i in range(len(d)):
    #     cv2.circle(image, (d[i], e[i]), 3, (255, 255, 0), -1)
    # for i in range(len(f)):
    #     cv2.circle(image, (f[i], g[i]), 3, (255, 0, 0), -1)
    # cv2.imshow('mask', image)
    # cv2.waitKey(0)
    from Matrix_generator import matrix_generator
    (matrix1, matrix_x1, matrix_y1) = matrix_generator(x, y, e, d)

    from Distances import mean_distance
    media_coxa1 = mean_distance(matrix_x1, matrix1)

    return media_coxa1
