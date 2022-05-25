# ################# Pixellib packages ##########################
import pixellib
from pixellib.torchbackend.instance import instanceSegmentation
import cv2
import numpy as np
import imutils

# ################# Mediapipe packages ##########################
import mediapipe as mp
import cv2
import math

# Adjust contrast image
# image = cv2.addWeighted(image, 2.5, np.zeros(image.shape, image.dtype), 0, 0)
# cv2.imshow('image', image)
# cv2.waitKey(0)

# dowload the pre-trained ckpt for image matting
pretrained_ckpt = 'D:/Documentos/ACRM/Alejandra/test_project/modnet_photographic_portrait_matting.ckpt'

# INFERENCE MATTE
# Install packages
import pandas as pd
import os
import sys
import argparse
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

from MODNet.src.models.modnet import MODNet

# define hyper-parameters
ref_size = 512

# define image to tensor transform
im_transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]
)

# create MODNet and load the pre-trained ckpt
modnet = MODNet(backbone_pretrained=False)
modnet = nn.DataParallel(modnet)

if torch.cuda.is_available():
    modnet = modnet.cuda()
    weights = torch.load('D:/Documentos/ACRM/Alejandra/test_project/modnet_photographic_portrait_matting.ckpt')
else:
    weights = torch.load('D:/Documentos/ACRM/Alejandra/test_project/modnet_photographic_portrait_matting.ckpt',
                         map_location=torch.device('gpu'))  # cambie 'cpu' por 'gpu' (11/05/22)

modnet.load_state_dict(weights)
modnet.eval()

# Calculate landmarks in frontal position
def Calculate_landmarks_frontal(d_input, d_output, d_landmark):
    # inference images
    from output_generator import output_generator
    # Extract list of objects inside output
    output_list = output_generator(modnet, ref_size, d_input, d_output, im_transform)
    # Create column names and dataframe
    column_name = ['nose', 'left shoulder', 'right shoulder', 'left elbow', 'right elbow', 'left wrist', 'right wrist',
                   'left knee', 'right knee', 'left ankle', 'right ankle']
    df2 = pd.DataFrame()
    for im_output in output_list:
        print('Process output image: {0}'.format(im_output))
        # read image
        mask2 = cv2.imread(os.path.join(d_output, im_output))
        mask_output = cv2.cvtColor(mask2, cv2.COLOR_BGR2GRAY)

        # find contours in threshold image, then grab the largest one
        cnts = cv2.findContours(mask_output, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        c = max(cnts, key=cv2.contourArea)

        # Extract output file name
        out_name, out_ext = os.path.splitext(os.path.basename(os.path.join(d_output, im_output)))
        # Extract list of objects inside input
        im_names2 = os.listdir(d_input)

        for im_name2 in im_names2:
            # Extract input file name
            inp_name, inp_ext = os.path.splitext(os.path.basename(os.path.join(d_input, im_name2)))
            if out_name == inp_name:
                im_name2 = cv2.imread(os.path.join(d_input, im_name2))
                image_contour = im_name2.copy()
                cv2.drawContours(image_contour, [c], -1, (0, 255, 255), 2)

                mp_pose = mp.solutions.pose
                mp_drawing = mp.solutions.drawing_utils
                mp_drawing_styles = mp.solutions.drawing_styles

                image2 = image_contour.copy()

                with mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5,
                                  model_complexity=2) as pose:
                    # Convert the BGR image to RGB and process it width MediaPipe Pose
                    results = pose.process(cv2.cvtColor(image2, cv2.COLOR_BGR2RGB))
                    # Extract height and widht
                    image_height, image_width, _ = image2.shape
                    # Print pose landmarks
                    x = []
                    y = []

                    x_lmark = [results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].x,
                               results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].x,
                               results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].x,
                               results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW].x,
                               results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW].x,
                               results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].x,
                               results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST].x,
                               results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE].x,
                               results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE].x,
                               results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE].x,
                               results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE].x]

                    y_lmark = [results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].y,
                               results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].y,
                               results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].y,
                               results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW].y,
                               results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW].y,
                               results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].y,
                               results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST].y,
                               results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE].y,
                               results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE].y,
                               results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE].y,
                               results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE].y]

                    if results.pose_landmarks is not None:
                        for i in range(len(x_lmark)):
                            x.append(int(x_lmark[i] * image_width))
                            y.append(int(y_lmark[i] * image_height))

                        for t in range(len(x)):
                            cv2.circle(image2, (int(x[t]), int(y[t])), 3, (0, 0, 255), -1)

                        x_contour = c[:, :, 0]
                        y_contour = c[:, :, 1]
                        # create empty arrays to save landmark extern points
                        d = []
                        e = []
                        for i in range(len(c)):
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

                        from Matrix_generator import matrix_generator
                        (matrix, matrix_x, matrix_y) = matrix_generator(x, y, e, d)

                        from Distances import mean_distance
                        media = mean_distance(matrix_x, matrix)

                        # extreme points for height
                        extTop = tuple(c[c[:, :, 1].argmin()][0])
                        extBot = tuple(c[c[:, :, 1].argmax()][0])

                        from scipy.spatial import distance as dist
                        RefObj = dist.euclidean(extTop, extBot)

                        image3 = im_name2.copy()
                        for i in range(len(d)):
                            cv2.circle(image3, (d[i], e[i]), 3, (128, 0, 0), -1)

                        cv2.circle(image3, extTop, 3, (255, 0, 0), -1)  # blue
                        cv2.circle(image3, extBot, 3, (255, 255, 0), -1)  # light blue

                        land_name = str(inp_name) + '_landmark' + '.jpeg'
                        cv2.imwrite(os.path.join(d_landmark, land_name), image3)

                        from braco_relax import braco_relaxado
                        b_relax_right = braco_relaxado(image2, mask2, x, y)

                        from cervical import perimetro_cervical
                        cerv = perimetro_cervical(image2, mask2, x, y)

                        # Extract photo name to insert in df
                        row_name = [inp_name]
                        df = pd.DataFrame([media], columns=column_name, index=row_name)
                        # Insert height
                        df = df.assign(height=[RefObj])
                        # Inser braco relaxado medio
                        df = df.assign(braco_relaxado_right=b_relax_right)
                        # Insert cervical
                        df = df.assign(cervical=cerv)
        df2 = df2.append(df)
    print(df2)
    df2.to_csv('Frontal.csv')

# Calculate landmarks in costa position
def Calculate_landmarks_costa(d_input, d_output, d_landmark):
    # inference images
    from output_generator import output_generator
    # Extract list of objects inside output
    output_list = output_generator(modnet, ref_size, d_input, d_output, im_transform)
    # Create column names and dataframe
    column_name = ['nose', 'left shoulder', 'right shoulder', 'left elbow', 'right elbow', 'left wrist', 'right wrist',
                   'left knee', 'right knee', 'left ankle', 'right ankle']
    df2 = pd.DataFrame()
    for im_output in output_list:
        print('Process output image: {0}'.format(im_output))
        # read image
        mask2 = cv2.imread(os.path.join(d_output, im_output))
        mask_output = cv2.cvtColor(mask2, cv2.COLOR_BGR2GRAY)

        # find contours in thresholded image, then grab the largest one
        cnts = cv2.findContours(mask_output, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        c = max(cnts, key=cv2.contourArea)

        # Extract output file name
        out_name, out_ext = os.path.splitext(os.path.basename(os.path.join(d_output, im_output)))
        # Extract list of objects inside input
        im_names2 = os.listdir(d_input)

        for im_name2 in im_names2:
            # Extract input file name
            inp_name, inp_ext = os.path.splitext(os.path.basename(os.path.join(d_input, im_name2)))
            if out_name == inp_name:
                im_name2 = cv2.imread(os.path.join(d_input, im_name2))
                image_contour = im_name2.copy()
                cv2.drawContours(image_contour, [c], -1, (0, 255, 255), 2)

                mp_pose = mp.solutions.pose
                mp_drawing = mp.solutions.drawing_utils
                mp_drawing_styles = mp.solutions.drawing_styles

                image2 = image_contour.copy()

                with mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5,
                                  model_complexity=2) as pose:
                    # Convert the BGR image to RGB and process it width MediaPipe Pose
                    results = pose.process(cv2.cvtColor(image2, cv2.COLOR_BGR2RGB))
                    # Extract height and widht
                    image_height, image_width, _ = image2.shape
                    # Print pose landmarks
                    x = []
                    y = []

                    x_lmark = [results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].x,
                               results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].x,
                               results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].x,
                               results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW].x,
                               results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW].x,
                               results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].x,
                               results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST].x,
                               results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE].x,
                               results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE].x,
                               results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE].x,
                               results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE].x]

                    y_lmark = [results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].y,
                               results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].y,
                               results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].y,
                               results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW].y,
                               results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW].y,
                               results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].y,
                               results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST].y,
                               results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE].y,
                               results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE].y,
                               results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE].y,
                               results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE].y]

                    if results.pose_landmarks is not None:
                        for i in range(len(x_lmark)):
                            x.append(int(x_lmark[i] * image_width))
                            y.append(int(y_lmark[i] * image_height))

                        for t in range(len(x)):
                            cv2.circle(image2, (int(x[t]), int(y[t])), 3, (0, 0, 255), -1)

                        x_contour = c[:, :, 0]
                        y_contour = c[:, :, 1]

                        # create empty arrays to save landmark extern points
                        d = []
                        e = []

                        for i in range(len(c)):
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

                        from Matrix_generator import matrix_generator
                        (matrix, matrix_x, matrix_y) = matrix_generator(x, y, e, d)

                        from Distances import mean_distance
                        media = mean_distance(matrix_x, matrix)

                        # extreme points for height
                        extTop = tuple(c[c[:, :, 1].argmin()][0])
                        extBot = tuple(c[c[:, :, 1].argmax()][0])

                        from scipy.spatial import distance as dist
                        RefObj = dist.euclidean(extTop, extBot)

                        from antebraco import antebraco
                        antebr = antebraco(image2, mask2, x, y)

                        image3 = im_name2.copy()
                        for i in range(len(d)):
                            cv2.circle(image3, (d[i], e[i]), 3, (128, 0, 0), -1)

                        cv2.circle(image3, extTop, 3, (255, 0, 0), -1)  # blue
                        cv2.circle(image3, extBot, 3, (255, 255, 0), -1)  # light blue

                        land_name = str(inp_name) + '_landmark' + '.jpeg'
                        cv2.imwrite(os.path.join(d_landmark, land_name), image3)
                        # Extract photo name to insert in df
                        row_name = [inp_name]
                        df = pd.DataFrame([media], columns=column_name, index=row_name)
                        # Insert height
                        df = df.assign(height=[RefObj])
                        # Insert antebraco
                        df = df.assign(antebraco=antebr)
        df2 = df2.append(df)
        print(df2)
    df2.to_csv('Costa.csv')


# Calculate landmarks in frontal position with cross arms
def Calculate_landmarks_frontal_cruz(d_input, d_output, d_landmark):
    # inference images
    from output_generator import output_generator
    # Extract list of objects inside output
    output_list = output_generator(modnet, ref_size, d_input, d_output, im_transform)
    # Create column names and dataframe
    column_name = ['nose', 'left shoulder', 'right shoulder', 'left hip', 'right hip', 'left knee', 'right knee',
                   'left ankle', 'right ankle']
    df2 = pd.DataFrame()
    for im_output in output_list:
        print('Process output image: {0}'.format(im_output))
        # read image
        mask2 = cv2.imread(os.path.join(d_output, im_output))
        mask_output = cv2.cvtColor(mask2, cv2.COLOR_BGR2GRAY)

        # find contours in thresholded image, then grab the largest one
        cnts = cv2.findContours(mask_output, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        c = max(cnts, key=cv2.contourArea)

        # Extract output file name
        out_name, out_ext = os.path.splitext(os.path.basename(os.path.join(d_output, im_output)))
        # Extract list of objects inside input
        im_names2 = os.listdir(d_input)

        for im_name2 in im_names2:
            # Extract input file name
            inp_name, inp_ext = os.path.splitext(os.path.basename(os.path.join(d_input, im_name2)))
            if out_name == inp_name:
                im_name2 = cv2.imread(os.path.join(d_input, im_name2))
                image_contour = im_name2.copy()
                cv2.drawContours(image_contour, [c], -1, (0, 255, 255), 2)

                mp_pose = mp.solutions.pose
                mp_drawing = mp.solutions.drawing_utils
                mp_drawing_styles = mp.solutions.drawing_styles
                image2 = image_contour.copy()

                with mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5,
                                  model_complexity=2) as pose:
                    # Convert the BGR image to RGB and process it width MediaPipe Pose
                    results = pose.process(cv2.cvtColor(image2, cv2.COLOR_BGR2RGB))
                    # Extract height and widht
                    image_height, image_width, _ = image2.shape
                    # Print pose landmarks
                    x = []
                    y = []

                    x_lmark = [results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].x,
                               results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].x,
                               results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].x,
                               results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].x,
                               results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP].x,
                               results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE].x,
                               results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE].x,
                               results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE].x,
                               results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE].x]

                    y_lmark = [results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].y,
                               results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].y,
                               results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].y,
                               results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].y,
                               results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP].y,
                               results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE].y,
                               results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE].y,
                               results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE].y,
                               results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE].y]

                    if results.pose_landmarks is not None:
                        for i in range(len(x_lmark)):
                            x.append(int(x_lmark[i] * image_width))
                            y.append(int(y_lmark[i] * image_height))

                        for t in range(len(x)):
                            cv2.circle(image2, (int(x[t]), int(y[t])), 3, (0, 0, 255), -1)

                        x_contour = c[:, :, 0]
                        y_contour = c[:, :, 1]

                        # create empty arrays to save landmark extern points
                        d = []
                        e = []

                        for i in range(len(c)):
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

                        from Matrix_generator import matrix_generator
                        (matrix, matrix_x, matrix_y) = matrix_generator(x, y, e, d)

                        from Distances import mean_distance
                        media = mean_distance(matrix_x, matrix)
                        # print(media)

                        # extreme points for height
                        extTop = tuple(c[c[:, :, 1].argmin()][0])
                        extBot = tuple(c[c[:, :, 1].argmax()][0])

                        from scipy.spatial import distance as dist
                        RefObj = dist.euclidean(extTop, extBot)

                        image3 = im_name2.copy()
                        for i in range(len(d)):
                            cv2.circle(image3, (d[i], e[i]), 3, (128, 0, 0), -1)

                        cv2.circle(image3, extTop, 3, (255, 0, 0), -1)  # blue
                        cv2.circle(image3, extBot, 3, (255, 255, 0), -1)  # light blue

                        land_name = str(inp_name) + '_landmark' + '.jpeg'
                        cv2.imwrite(os.path.join(d_landmark, land_name), image3)

                        from coxa_media import coxa_media
                        cox_media1, cox_media2 = coxa_media(im_name2, x_lmark, y_lmark, image_width, image_height, x_contour, y_contour, c)

                        from crop_torax import cint_minima
                        cint_min = cint_minima(im_name2, mask2, x_lmark, y_lmark, image_width, image_height)

                        from panturilha import pantu_max
                        pantur_max = pantu_max(im_name2, mask2, x, y)

                        # Extract photo name to insert in df
                        row_name = [inp_name]
                        df = pd.DataFrame([media], columns=column_name, index=row_name)
                        # Insert height
                        df = df.assign(height=[RefObj])
                        # Insert coxa media
                        df = df.assign(coxa_media_right=cox_media1)
                        # Insert cintura minima
                        df = df.assign(cintura_minima=cint_min)
                        # Insert cintura minima
                        df = df.assign(panturilha_maxima=pantur_max)
        df2 = df2.append(df)
    print(df2)
    df2.to_csv('Frontal_cruzado.csv')


# Calculate landmarks in right lateral position
def Calculate_landmarks_lateral1(d_input, d_output, d_landmark):
    # inference images
    from output_generator import output_generator
    # Extract list of objects inside output
    output_list = output_generator(modnet, ref_size, d_input, d_output, im_transform)
    # Create column names and dataframe
    column_name = ['nose', 'right shoulder', 'right hip', 'right knee']
    df2 = pd.DataFrame()
    for im_output in output_list:
        print('Process output image: {0}'.format(im_output))
        # read image
        mask2 = cv2.imread(os.path.join(d_output, im_output))
        mask_output = cv2.cvtColor(mask2, cv2.COLOR_BGR2GRAY)

        # find contours in thresholded image, then grab the largest one
        cnts = cv2.findContours(mask_output, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        c = max(cnts, key=cv2.contourArea)

        # Extract output file name
        out_name, out_ext = os.path.splitext(os.path.basename(os.path.join(d_output, im_output)))
        # Extract list of objects inside input
        im_names2 = os.listdir(d_input)

        for im_name2 in im_names2:
            # Extract input file name
            inp_name, inp_ext = os.path.splitext(os.path.basename(os.path.join(d_input, im_name2)))
            if out_name == inp_name:
                im_name2 = cv2.imread(os.path.join(d_input, im_name2))
                image_contour = im_name2.copy()
                cv2.drawContours(image_contour, [c], -1, (0, 255, 255), 2)

                mp_pose = mp.solutions.pose
                mp_drawing = mp.solutions.drawing_utils
                mp_drawing_styles = mp.solutions.drawing_styles
                image2 = image_contour.copy()

                with mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5,
                                  model_complexity=2) as pose:
                    # Convert the BGR image to RGB and process it width MediaPipe Pose
                    results = pose.process(cv2.cvtColor(image2, cv2.COLOR_BGR2RGB))
                    # Extract height and widht
                    image_height, image_width, _ = image2.shape
                    # Print pose landmarks
                    x = []
                    y = []

                    x_lmark = [results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].x,
                               results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].x,
                               results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP].x,
                               results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE].x]
                    y_lmark = [results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].y,
                               results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].y,
                               results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP].y,
                               results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE].y]

                    if results.pose_landmarks is not None:
                        for i in range(len(x_lmark)):
                            x.append(int(x_lmark[i] * image_width))
                            y.append(int(y_lmark[i] * image_height))

                        for t in range(len(x)):
                            cv2.circle(image2, (int(x[t]), int(y[t])), 3, (0, 0, 255), -1)
                        x_contour = c[:, :, 0]
                        y_contour = c[:, :, 1]

                        # create empty arrays to save landmark extern points
                        d = []
                        e = []
                        # find points in contour
                        for i in range(len(y)):
                            for j in range(len(c)):
                                if y[i] == y_contour[j][0]:
                                    d.append(x_contour[j][0])
                                    e.append(y_contour[j][0])
                                if y[i] == y_contour[j][0] + 1:
                                    d.append(x_contour[j][0])
                                    e.append(y_contour[j][0])
                                if y[i] == y_contour[j][0] - 1:
                                    d.append(x_contour[j][0])
                                    e.append(y_contour[j][0])
                        # create matrix that save coord points of contour that match each body part
                        from Matrix_generator import matrix_generator
                        (matrix, matrix_x, matrix_y) = matrix_generator(x, y, e, d)

                        # Calculate mean distance between points
                        from Distances import mean_distance
                        media = mean_distance(matrix_x, matrix)

                        # Calculate cervical
                        from cervical_lateral import perimetro_cervical_lat1
                        # perimetro_cervical_lat1(image2, mask2, x, y)

                        # extreme points for height
                        extTop = tuple(c[c[:, :, 1].argmin()][0])
                        extBot = tuple(c[c[:, :, 1].argmax()][0])

                        from scipy.spatial import distance as dist
                        RefObj = dist.euclidean(extTop, extBot)

                        image3 = im_name2.copy()
                        for i in range(len(d)):
                            cv2.circle(image3, (d[i], e[i]), 3, (255, 255, 0), -1)

                        cv2.circle(image3, extTop, 5, (255, 0, 0), -1)  # blue
                        cv2.circle(image3, extBot, 5, (255, 255, 0), -1)  # light blue

                        land_name = str(inp_name) + '_landmark' + '.jpeg'
                        cv2.imwrite(os.path.join(d_landmark, land_name), image3)
                        # Extract photo name to insert in df
                        row_name = [inp_name]
                        df = pd.DataFrame([media], columns=column_name, index=row_name)
                        # Insert height
                        df = df.assign(height=[RefObj])
        df2 = df2.append(df)
        print(df2)
    df2.to_csv('Lateral_direito.csv')


# Calculate landmarks in left lateral position
def Calculate_landmarks_lateral2(d_input, d_output, d_landmark):
    # inference images
    from output_generator import output_generator
    # Extract list of objects inside output
    output_list = output_generator(modnet, ref_size, d_input, d_output, im_transform)
    column_name = ['nose', 'left shoulder', 'left hip', 'left knee']
    df2 = pd.DataFrame()
    for im_output in output_list:
        print('Process output image: {0}'.format(im_output))
        # read image
        mask2 = cv2.imread(os.path.join(d_output, im_output))
        mask_output = cv2.cvtColor(mask2, cv2.COLOR_BGR2GRAY)

        # find contours in thresholded image, then grab the largest one
        cnts = cv2.findContours(mask_output, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        c = max(cnts, key=cv2.contourArea)

        # Extract output file name
        out_name, out_ext = os.path.splitext(os.path.basename(os.path.join(d_output, im_output)))
        # Extract list of objects inside input
        im_names2 = os.listdir(d_input)

        for im_name2 in im_names2:
            # Extract input file name
            inp_name, inp_ext = os.path.splitext(os.path.basename(os.path.join(d_input, im_name2)))
            if out_name == inp_name:
                im_name2 = cv2.imread(os.path.join(d_input, im_name2))
                image_contour = im_name2.copy()
                cv2.drawContours(image_contour, [c], -1, (0, 255, 255), 2)

                mp_pose = mp.solutions.pose
                mp_drawing = mp.solutions.drawing_utils
                mp_drawing_styles = mp.solutions.drawing_styles

                image2 = image_contour.copy()

                with mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5,
                                  model_complexity=2) as pose:
                    # Convert the BGR image to RGB and process it width MediaPipe Pose
                    results = pose.process(cv2.cvtColor(image2, cv2.COLOR_BGR2RGB))
                    # Extract height and widht
                    image_height, image_width, _ = image2.shape
                    # Print pose landmarks
                    x = []
                    y = []

                    x_lmark = [results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].x,
                               results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].x,
                               results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].x,
                               results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE].x]

                    y_lmark = [results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].y,
                               results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].y,
                               results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].y,
                               results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE].y]

                    if results.pose_landmarks is not None:
                        for i in range(len(x_lmark)):
                            x.append(int(x_lmark[i] * image_width))
                            y.append(int(y_lmark[i] * image_height))

                        for t in range(len(x)):
                            cv2.circle(image2, (int(x[t]), int(y[t])), 3, (0, 0, 255), -1)

                        x_contour = c[:, :, 0]
                        y_contour = c[:, :, 1]

                        # create empty arrays to save landmark extern points
                        d = []
                        e = []

                        for i in range(len(c)):
                            for j in range(len(y)):
                                if y_contour[i][0] == y[j]:
                                    d.append(x_contour[i][0])
                                    e.append(y_contour[i][0])
                                elif y_contour[i][0] + 1 == y[j]:
                                    d.append(x_contour[i][0])
                                    e.append(y_contour[i][0])
                                elif y_contour[i][0] - 1 == y[j]:
                                    d.append(x_contour[i][0])
                                    e.append(y_contour[i][0])

                        from Matrix_generator import matrix_generator
                        (matrix, matrix_x, matrix_y) = matrix_generator(x, y, e, d)

                        from Distances import mean_distance
                        media = mean_distance(matrix_x, matrix)
                        print(media)

                        # extreme points for height
                        extTop = tuple(c[c[:, :, 1].argmin()][0])
                        extBot = tuple(c[c[:, :, 1].argmax()][0])

                        from scipy.spatial import distance as dist
                        RefObj = dist.euclidean(extTop, extBot)

                        image3 = im_name2.copy()
                        for i in range(len(d)):
                            cv2.circle(image3, (d[i], e[i]), 3, (128, 0, 0), -1)

                        cv2.circle(image3, extTop, 5, (255, 0, 0), -1)  # blue
                        cv2.circle(image3, extBot, 5, (255, 255, 0), -1)  # light blue

                        land_name = str(inp_name) + '_landmark' + '.jpeg'
                        cv2.imwrite(os.path.join(d_landmark, land_name), image3)
                        # Extract photo name to insert in df
                        row_name = [inp_name]
                        df = pd.DataFrame([media], columns=column_name, index=row_name)
                        # Insert height
                        df = df.assign(height=[RefObj])
        df2 = df2.append(df)
        print(df2)
    df2.to_csv('Lateral_esquerdo.csv')


frontal_input_directory = 'D:/Documentos/ACRM/Alejandra/test_project/Amostras/Frontal'
frontal_output_directory = 'D:/Documentos/ACRM/Alejandra/test_project/output/Frontal'
frontal_landmark_directory = 'D:/Documentos/ACRM/Alejandra/test_project/landmarks/Frontal'

costa_input_directory = 'D:/Documentos/ACRM/Alejandra/test_project/Amostras/Costa'
costa_output_directory = 'D:/Documentos/ACRM/Alejandra/test_project/output/Costa'
costa_landmark_directory = 'D:/Documentos/ACRM/Alejandra/test_project/landmarks/Costa'

frontal_cross_input_directory = 'D:/Documentos/ACRM/Alejandra/test_project/Amostras/Braco_cruzado'
frontal_cross_output_directory = 'D:/Documentos/ACRM/Alejandra/test_project/output/Braco_cruzado'
frontal_cross_landmark_directory = 'D:/Documentos/ACRM/Alejandra/test_project/landmarks/Braco_cruzado'

lateral1_input_directory = 'D:/Documentos/ACRM/Alejandra/test_project/Amostras/Lat_Direita'
lateral1_output_directory = 'D:/Documentos/ACRM/Alejandra/test_project/output/Lat_Direita'
lateral1_landmark_directory = 'D:/Documentos/ACRM/Alejandra/test_project/landmarks/Lat_Direita'

lateral2_input_directory = 'D:/Documentos/ACRM/Alejandra/test_project/Amostras/Lat_Esquerda'
lateral2_output_directory = 'D:/Documentos/ACRM/Alejandra/test_project/output/Lat_Esquerda'
lateral2_landmark_directory = 'D:/Documentos/ACRM/Alejandra/test_project/landmarks/Lat_Esquerda'

# Calculate_landmarks_frontal(frontal_input_directory, frontal_output_directory, frontal_landmark_directory)
# Calculate_landmarks_frontal_cruz(frontal_cross_input_directory, frontal_cross_output_directory,
#                                  frontal_cross_landmark_directory)
# Calculate_landmarks_costa(costa_input_directory, costa_output_directory, costa_landmark_directory)
Calculate_landmarks_lateral1(lateral1_input_directory, lateral1_output_directory, lateral1_landmark_directory)
# Calculate_landmarks_lateral2(lateral2_input_directory, lateral2_output_directory, lateral2_landmark_directory)
