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

# Read the image
# image = cv2.imread("D:/Documentos/ACRM/Alejandra/test_project/input/test11.jpeg")
# Show the image
# cv2.imshow('image', image)
# cv2.waitKey(0)

# Adjust contrast image
# image = cv2.addWeighted(image, 2.5, np.zeros(image.shape, image.dtype), 0, 0)
# cv2.imshow('image', image)
# cv2.waitKey(0)

# dowload the pre-trained ckpt for image matting
pretrained_ckpt = 'D:/Documentos/ACRM/Alejandra/test_project/modnet_photographic_portrait_matting.ckpt'

# INFERENCE MATTE
# Install packages
import os
import sys
import argparse
import numpy as np
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
    im_names = os.listdir(d_input)
    for im_name in im_names:
        print('Process image: {0}'.format(im_name))

        # read image
        im = Image.open(os.path.join(d_input, im_name))

        # unify image channels to 3
        im = np.asarray(im)
        if len(im.shape) == 2:
            im = im[:, :, None]
        if im.shape[2] == 1:
            im = np.repeat(im, 3, axis=2)
        elif im.shape[2] == 4:
            im = im[:, :, 0:3]

        # convert image to PyTorch tensor
        im = Image.fromarray(im)
        im = im_transform(im)

        # add mini-batch dim
        im = im[None, :, :, :]

        # resize image for input
        im_b, im_c, im_h, im_w = im.shape
        if max(im_h, im_w) < ref_size or min(im_h, im_w) > ref_size:
            if im_w >= im_h:
                im_rh = ref_size
                im_rw = int(im_w / im_h * ref_size)
            elif im_w < im_h:
                im_rw = ref_size
                im_rh = int(im_h / im_w * ref_size)
        else:
            im_rh = im_h
            im_rw = im_w

        im_rw = im_rw - im_rw % 64
        im_rh = im_rh - im_rh % 64
        im = F.interpolate(im, size=(im_rh, im_rw), mode='area')

        # inference
        _, _, matte = modnet(im.cuda() if torch.cuda.is_available() else im, True)

        # resize and save matte
        matte = F.interpolate(matte, size=(im_h, im_w), mode='area')
        matte = matte[0][0].data.cpu().numpy()
        matte_name = im_name.split('.')[0] + '.png'
        Image.fromarray(((matte * 255).astype('uint8')), mode='L').save(
            os.path.join(d_output, matte_name))
    # Extract list of objects inside output
    output_list = os.listdir(d_output)
    for im_output in output_list:
        print('Process output image: {0}'.format(im_output))
        # read image
        mask2 = cv2.imread(os.path.join(d_output, im_output))
        mask_output = cv2.cvtColor(mask2, cv2.COLOR_BGR2GRAY)
        # cv2.imshow('Mask2', mask_output)
        # cv2.waitKey(0)

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
                # cv2.imshow('Contour image', image_contour)
                # cv2.waitKey(0)

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
                               results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE].x,
                               results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE].x,
                               results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE].x,
                               results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE].x,
                               results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].x,
                               results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].x,
                               results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW].x,
                               results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW].x,
                               results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].x,
                               results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST].x]

                    y_lmark = [results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].y,
                               results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE].y,
                               results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE].y,
                               results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE].y,
                               results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE].y,
                               results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].y,
                               results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].y,
                               results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW].y,
                               results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW].y,
                               results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].y,
                               results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST].y]

                    if results.pose_landmarks is not None:
                        for i in range(len(x_lmark)):
                            x.append(int(x_lmark[i] * image_width))
                            y.append(int(y_lmark[i] * image_height))

                        for t in range(len(x)):
                            cv2.circle(image2, (int(x[t]), int(y[t])), 3, (0, 0, 255), -1)
                        # cv2.imshow('Image landmarks', image2)
                        # cv2.waitKey(0)

                        x_contour = c[:, :, 0]
                        y_contour = c[:, :, 1]

                        # create empty arrays to save landmark extern points
                        d = []
                        e = []
                        # f = []
                        # g = []
                        # h = []
                        # k = []

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

                        # extreme points for height
                        extTop = tuple(c[c[:, :, 1].argmin()][0])
                        extBot = tuple(c[c[:, :, 1].argmax()][0])

                        image3 = im_name2.copy()
                        # image4 = im_name2.copy()
                        # image5 = im_name2.copy()
                        for i in range(len(d)):
                            cv2.circle(image3, (d[i], e[i]), 3, (128, 0, 0), -1)

                        cv2.circle(image3, extTop, 3, (255, 0, 0), -1)  # blue
                        cv2.circle(image3, extBot, 3, (255, 255, 0), -1)  # light blue

                        # for i in range(len(f)):
                        #    cv2.circle(image3, (f[i], g[i]), 5, (128, 0, 0), -1)

                        # for i in range(len(h)):
                        #    cv2.circle(image3, (h[i], k[i]), 5, (128, 0, 0), -1)
                        # cv2.imshow('Image4', image4)
                        # cv2.waitKey(0)
                        land_name = str(inp_name) + '_landmark' + '.jpeg'
                        # land_name_2 = str(inp_name) + '_landmark_2' + '.jpeg'
                        # land_name_3 = str(inp_name) + '_landmark_3' + '.jpeg'
                        cv2.imwrite(os.path.join(d_landmark, land_name), image3)
                        # cv2.imwrite(os.path.join(d_landmark, land_name_2), image4)
                        # cv2.imwrite(os.path.join(d_landmark, land_name_3), image5)
                        # cv2.imshow('Image3', image3)
                        # cv2.waitKey(0)


# Calculate landmarks in costa position
def Calculate_landmarks_costa(d_input, d_output, d_landmark):
    # inference images
    im_names = os.listdir(d_input)
    for im_name in im_names:
        print('Process image: {0}'.format(im_name))

        # read image
        im = Image.open(os.path.join(d_input, im_name))

        # unify image channels to 3
        im = np.asarray(im)
        if len(im.shape) == 2:
            im = im[:, :, None]
        if im.shape[2] == 1:
            im = np.repeat(im, 3, axis=2)
        elif im.shape[2] == 4:
            im = im[:, :, 0:3]

        # convert image to PyTorch tensor
        im = Image.fromarray(im)
        im = im_transform(im)

        # add mini-batch dim
        im = im[None, :, :, :]

        # resize image for input
        im_b, im_c, im_h, im_w = im.shape
        if max(im_h, im_w) < ref_size or min(im_h, im_w) > ref_size:
            if im_w >= im_h:
                im_rh = ref_size
                im_rw = int(im_w / im_h * ref_size)
            elif im_w < im_h:
                im_rw = ref_size
                im_rh = int(im_h / im_w * ref_size)
        else:
            im_rh = im_h
            im_rw = im_w

        im_rw = im_rw - im_rw % 64
        im_rh = im_rh - im_rh % 64
        im = F.interpolate(im, size=(im_rh, im_rw), mode='area')

        # inference
        _, _, matte = modnet(im.cuda() if torch.cuda.is_available() else im, True)

        # resize and save matte
        matte = F.interpolate(matte, size=(im_h, im_w), mode='area')
        matte = matte[0][0].data.cpu().numpy()
        matte_name = im_name.split('.')[0] + '.png'
        Image.fromarray(((matte * 255).astype('uint8')), mode='L').save(
            os.path.join(d_output, matte_name))
    # Extract list of objects inside output
    output_list = os.listdir(d_output)
    for im_output in output_list:
        print('Process output image: {0}'.format(im_output))
        # read image
        mask2 = cv2.imread(os.path.join(d_output, im_output))
        mask_output = cv2.cvtColor(mask2, cv2.COLOR_BGR2GRAY)
        # cv2.imshow('Mask2', mask_output)
        # cv2.waitKey(0)

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
                # cv2.imshow('Contour image', image_contour)
                # cv2.waitKey(0)

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
                               results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE].x,
                               results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE].x,
                               results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE].x,
                               results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE].x,
                               results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].x,
                               results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].x,
                               results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW].x,
                               results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW].x,
                               results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].x,
                               results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST].x]

                    y_lmark = [results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].y,
                               results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE].y,
                               results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE].y,
                               results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE].y,
                               results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE].y,
                               results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].y,
                               results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].y,
                               results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW].y,
                               results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW].y,
                               results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].y,
                               results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST].y]

                    if results.pose_landmarks is not None:
                        for i in range(len(x_lmark)):
                            x.append(int(x_lmark[i] * image_width))
                            y.append(int(y_lmark[i] * image_height))

                        for t in range(len(x)):
                            cv2.circle(image2, (int(x[t]), int(y[t])), 3, (0, 0, 255), -1)

                        # cv2.imshow('Image landmarks', image2)
                        # cv2.waitKey(0)
                        x_contour = c[:, :, 0]
                        y_contour = c[:, :, 1]

                        # create empty arrays to save landmark extern points
                        d = []
                        e = []
                        # f = []
                        # g = []
                        # h = []
                        # k = []
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

                        # extreme points for height
                        extTop = tuple(c[c[:, :, 1].argmin()][0])
                        extBot = tuple(c[c[:, :, 1].argmax()][0])

                        image3 = im_name2.copy()
                        # image4 = im_name2.copy()
                        # image5 = im_name2.copy()

                        for i in range(len(d)):
                            cv2.circle(image3, (d[i], e[i]), 3, (128, 0, 0), -1)

                        cv2.circle(image3, extTop, 3, (255, 0, 0), -1)  # blue
                        cv2.circle(image3, extBot, 3, (255, 255, 0), -1)  # light blue

                        # cv2.imshow('Image4', image4)
                        # cv2.waitKey(0)

                        land_name = str(inp_name) + '_landmark' + '.jpeg'
                        # land_name_2 = str(inp_name) + '_landmark_2' + '.jpeg'
                        # land_name_3 = str(inp_name) + '_landmark_3' + '.jpeg'

                        cv2.imwrite(os.path.join(d_landmark, land_name), image3)
                        # cv2.imwrite(os.path.join(d_landmark, land_name_2), image4)
                        # cv2.imwrite(os.path.join(d_landmark, land_name_3), image5)
                        # cv2.imshow('Image3', image3)
                        # cv2.waitKey(0)


# Calculate landmarks in frontal position with cross arms
def Calculate_landmarks_frontal_cruz(d_input, d_output, d_landmark):
    # inference images
    im_names = os.listdir(d_input)
    for im_name in im_names:
        print('Process image: {0}'.format(im_name))

        # read image
        im = Image.open(os.path.join(d_input, im_name))

        # unify image channels to 3
        im = np.asarray(im)
        if len(im.shape) == 2:
            im = im[:, :, None]
        if im.shape[2] == 1:
            im = np.repeat(im, 3, axis=2)
        elif im.shape[2] == 4:
            im = im[:, :, 0:3]

        # convert image to PyTorch tensor
        im = Image.fromarray(im)
        im = im_transform(im)

        # add mini-batch dim
        im = im[None, :, :, :]

        # resize image for input
        im_b, im_c, im_h, im_w = im.shape
        if max(im_h, im_w) < ref_size or min(im_h, im_w) > ref_size:
            if im_w >= im_h:
                im_rh = ref_size
                im_rw = int(im_w / im_h * ref_size)
            elif im_w < im_h:
                im_rw = ref_size
                im_rh = int(im_h / im_w * ref_size)
        else:
            im_rh = im_h
            im_rw = im_w

        im_rw = im_rw - im_rw % 64
        im_rh = im_rh - im_rh % 64
        im = F.interpolate(im, size=(im_rh, im_rw), mode='area')

        # inference
        _, _, matte = modnet(im.cuda() if torch.cuda.is_available() else im, True)

        # resize and save matte
        matte = F.interpolate(matte, size=(im_h, im_w), mode='area')
        matte = matte[0][0].data.cpu().numpy()
        matte_name = im_name.split('.')[0] + '.png'
        Image.fromarray(((matte * 255).astype('uint8')), mode='L').save(
            os.path.join(d_output, matte_name))
    # Extract list of objects inside output
    output_list = os.listdir(d_output)
    for im_output in output_list:
        print('Process output image: {0}'.format(im_output))
        # read image
        mask2 = cv2.imread(os.path.join(d_output, im_output))
        mask_output = cv2.cvtColor(mask2, cv2.COLOR_BGR2GRAY)
        # cv2.imshow('Mask2', mask_output)
        # cv2.waitKey(0)

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
                # cv2.imshow('Contour image', image_contour)
                # cv2.waitKey(0)

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
                               results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP].x,
                               results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].x,
                               results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE].x,
                               results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE].x,
                               results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE].x,
                               results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE].x,
                               results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].x,
                               results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].x]

                    y_lmark = [results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].y,
                               results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP].y,
                               results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].y,
                               results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE].y,
                               results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE].y,
                               results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE].y,
                               results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE].y,
                               results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].y,
                               results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].y]

                    if results.pose_landmarks is not None:
                        for i in range(len(x_lmark)):
                            x.append(int(x_lmark[i] * image_width))
                            y.append(int(y_lmark[i] * image_height))

                        for t in range(len(x)):
                            cv2.circle(image2, (int(x[t]), int(y[t])), 3, (0, 0, 255), -1)

                        # cv2.imshow('Image landmarks', image2)
                        # cv2.waitKey(0)

                        x_contour = c[:, :, 0]
                        y_contour = c[:, :, 1]

                        # create empty arrays to save landmark extern points
                        d = []
                        e = []
                        # f = []
                        # g = []
                        # h = []
                        # k = []

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

                        # extreme points for height
                        extTop = tuple(c[c[:, :, 1].argmin()][0])
                        extBot = tuple(c[c[:, :, 1].argmax()][0])

                        image3 = im_name2.copy()
                        # image4 = im_name2.copy()
                        # image5 = im_name2.copy()

                        for i in range(len(d)):
                            cv2.circle(image3, (d[i], e[i]), 3, (128, 0, 0), -1)

                        cv2.circle(image3, extTop, 3, (255, 0, 0), -1)  # blue
                        cv2.circle(image3, extBot, 3, (255, 255, 0), -1)  # light blue

                        land_name = str(inp_name) + '_landmark' + '.jpeg'
                        cv2.imwrite(os.path.join(d_landmark, land_name), image3)

                        # cv2.imshow('Image3', image3)
                        # cv2.waitKey(0)


# Calculate landmarks in right lateral position
def Calculate_landmarks_lateral1(d_input, d_output, d_landmark):
    # inference images
    im_names = os.listdir(d_input)
    for im_name in im_names:
        print('Process image: {0}'.format(im_name))

        # read image
        im = Image.open(os.path.join(d_input, im_name))

        # unify image channels to 3
        im = np.asarray(im)
        if len(im.shape) == 2:
            im = im[:, :, None]
        if im.shape[2] == 1:
            im = np.repeat(im, 3, axis=2)
        elif im.shape[2] == 4:
            im = im[:, :, 0:3]

        # convert image to PyTorch tensor
        im = Image.fromarray(im)
        im = im_transform(im)

        # add mini-batch dim
        im = im[None, :, :, :]

        # resize image for input
        im_b, im_c, im_h, im_w = im.shape
        if max(im_h, im_w) < ref_size or min(im_h, im_w) > ref_size:
            if im_w >= im_h:
                im_rh = ref_size
                im_rw = int(im_w / im_h * ref_size)
            elif im_w < im_h:
                im_rw = ref_size
                im_rh = int(im_h / im_w * ref_size)
        else:
            im_rh = im_h
            im_rw = im_w

        im_rw = im_rw - im_rw % 64
        im_rh = im_rh - im_rh % 64
        im = F.interpolate(im, size=(im_rh, im_rw), mode='area')

        # inference
        _, _, matte = modnet(im.cuda() if torch.cuda.is_available() else im, True)

        # resize and save matte
        matte = F.interpolate(matte, size=(im_h, im_w), mode='area')
        matte = matte[0][0].data.cpu().numpy()
        matte_name = im_name.split('.')[0] + '.png'
        Image.fromarray(((matte * 255).astype('uint8')), mode='L').save(
            os.path.join(d_output, matte_name))
    # Extract list of objects inside output
    output_list = os.listdir(d_output)

    for im_output in output_list:
        print('Process output image: {0}'.format(im_output))
        # read image
        mask2 = cv2.imread(os.path.join(d_output, im_output))
        mask_output = cv2.cvtColor(mask2, cv2.COLOR_BGR2GRAY)
        # cv2.imshow('Mask2', mask_output)
        # cv2.waitKey(0)

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
                # cv2.imshow('Contour image', image_contour)
                # cv2.waitKey(0)
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

                        # cv2.imshow('Image landmarks', image2)
                        # cv2.waitKey(0)

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

                        # create matrix that save coord points of countor that match each body part
                        matrix_x = [[] for _ in range(len(x))]
                        matrix_y = [[] for _ in range(len(y))]
                        for i in range(len(y)):
                            for j in range(len(e)):
                                if y[i] == e[j] or y[i] + 1 == e[j] or y[i] - 1 == e[j]:
                                    matrix_y[i].append(e[j])
                                    matrix_x[i].append(d[j])
                        # create matrix with elements from matrix_x and matrix_y
                        matrix = [[] for _ in range(len(matrix_x))]
                        for i in range(len(matrix_x)):
                            for j in range(len(matrix_x[i])):
                                matrix[i].append((matrix_x[i][j], matrix_y[i][j]))

                        # Calculate mean distance between points
                        # import packages
                        from itertools import permutations
                        from scipy.spatial import distance as dist
                        # create landmark distance matrix
                        d_landmk = [[] for _ in range(len(matrix_x))]
                        # create array for mean distance
                        media = []
                        for i in range(len(matrix_x)):
                            if len(matrix[i]) != 0:
                                permut = permutations(matrix[i], 2)
                                b = list(permut)
                                n = len(matrix[i])
                                if n != 1:
                                    for k in range(n * (n - 1)):
                                        d_landmk[i].append(dist.euclidean(b[k][0], b[k][1]))
                                    arr = np.asarray(d_landmk[i]).reshape(n, n - 1)
                                    iu = np.triu_indices(n - 1)
                                    arr2 = []
                                    for j in arr[iu]:
                                        if j > 10:
                                            arr2.append(j)
                                    if not arr2 == []:
                                        a = np.mean(arr2)
                                        media.append(a)
                                    else:
                                        media.append(0)
                            else:
                                d_landmk[i].append(0)
                                media.append(0)

                        print(media)

                        # extreme points for height
                        extTop = tuple(c[c[:, :, 1].argmin()][0])
                        extBot = tuple(c[c[:, :, 1].argmax()][0])

                        image3 = im_name2.copy()
                        for i in range(len(d)):
                            cv2.circle(image3, (d[i], e[i]), 3, (255, 255, 0), -1)

                        cv2.circle(image3, extTop, 5, (255, 0, 0), -1)  # blue
                        cv2.circle(image3, extBot, 5, (255, 255, 0), -1)  # light blue

                        # cv2.imshow('Image4', image4)
                        # cv2.waitKey(0)

                        land_name = str(inp_name) + '_landmark' + '.jpeg'

                        cv2.imwrite(os.path.join(d_landmark, land_name), image3)
                        # cv2.imshow('Image3', image3)
                        # cv2.waitKey(0)


# Calculate landmarks in left lateral position
def Calculate_landmarks_lateral2(d_input, d_output, d_landmark):
    # inference images
    im_names = os.listdir(d_input)
    for im_name in im_names:
        print('Process image: {0}'.format(im_name))

        # read image
        im = Image.open(os.path.join(d_input, im_name))

        # unify image channels to 3
        im = np.asarray(im)
        if len(im.shape) == 2:
            im = im[:, :, None]
        if im.shape[2] == 1:
            im = np.repeat(im, 3, axis=2)
        elif im.shape[2] == 4:
            im = im[:, :, 0:3]

        # convert image to PyTorch tensor
        im = Image.fromarray(im)
        im = im_transform(im)

        # add mini-batch dim
        im = im[None, :, :, :]

        # resize image for input
        im_b, im_c, im_h, im_w = im.shape
        if max(im_h, im_w) < ref_size or min(im_h, im_w) > ref_size:
            if im_w >= im_h:
                im_rh = ref_size
                im_rw = int(im_w / im_h * ref_size)
            elif im_w < im_h:
                im_rw = ref_size
                im_rh = int(im_h / im_w * ref_size)
        else:
            im_rh = im_h
            im_rw = im_w

        im_rw = im_rw - im_rw % 64
        im_rh = im_rh - im_rh % 64
        im = F.interpolate(im, size=(im_rh, im_rw), mode='area')

        # inference
        _, _, matte = modnet(im.cuda() if torch.cuda.is_available() else im, True)

        # resize and save matte
        matte = F.interpolate(matte, size=(im_h, im_w), mode='area')
        matte = matte[0][0].data.cpu().numpy()
        matte_name = im_name.split('.')[0] + '.png'
        Image.fromarray(((matte * 255).astype('uint8')), mode='L').save(
            os.path.join(d_output, matte_name))
    # Extract list of objects inside output
    output_list = os.listdir(d_output)

    for im_output in output_list:
        print('Process output image: {0}'.format(im_output))
        # read image
        mask2 = cv2.imread(os.path.join(d_output, im_output))
        mask_output = cv2.cvtColor(mask2, cv2.COLOR_BGR2GRAY)
        # cv2.imshow('Mask2', mask_output)
        # cv2.waitKey(0)

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
                # cv2.imshow('Contour image', image_contour)
                # cv2.waitKey(0)

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
                               results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].x,
                               results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].x,
                               results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE].x]

                    y_lmark = [results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].y,
                               results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].y,
                               results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].y,
                               results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE].y]

                    if results.pose_landmarks is not None:
                        for i in range(len(x_lmark)):
                            x.append(int(x_lmark[i] * image_width))
                            y.append(int(y_lmark[i] * image_height))

                        for t in range(len(x)):
                            cv2.circle(image2, (int(x[t]), int(y[t])), 3, (0, 0, 255), -1)

                        # cv2.imshow('Image landmarks', image2)
                        # cv2.waitKey(0)

                        x_contour = c[:, :, 0]
                        y_contour = c[:, :, 1]

                        # create empty arrays to save landmark extern points
                        d = []
                        e = []
                        # f = []
                        # g = []
                        # h = []
                        # k = []

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

                        # extreme points for height
                        extTop = tuple(c[c[:, :, 1].argmin()][0])
                        extBot = tuple(c[c[:, :, 1].argmax()][0])

                        image3 = im_name2.copy()

                        for i in range(len(d)):
                            cv2.circle(image3, (d[i], e[i]), 3, (128, 0, 0), -1)

                        cv2.circle(image3, extTop, 5, (255, 0, 0), -1)  # blue
                        cv2.circle(image3, extBot, 5, (255, 255, 0), -1)  # light blue

                        # cv2.imshow('Image4', image4)
                        # cv2.waitKey(0)

                        land_name = str(inp_name) + '_landmark' + '.jpeg'

                        cv2.imwrite(os.path.join(d_landmark, land_name), image3)

                        # cv2.imshow('Image3', image3)
                        # cv2.waitKey(0)


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
#                                 frontal_cross_landmark_directory)
# Calculate_landmarks_costa(costa_input_directory, costa_output_directory, costa_landmark_directory)
Calculate_landmarks_lateral1(lateral1_input_directory, lateral1_output_directory, lateral1_landmark_directory)
# Calculate_landmarks_lateral2(lateral2_input_directory, lateral2_output_directory, lateral2_landmark_directory)
