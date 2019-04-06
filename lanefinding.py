#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 31 13:50:09 2019

@author: phongnd205
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
from os.path import join, basename
from ipykernel.kernelapp import IPKernelApp

def make_coordinates(image, line_parameters):
    slope, intercept = line_parameters
    y1 = image.shape[0]
    y2 = int(y1*(3/5))
    x1 = int((y1 - intercept)/slope)
    x2 = int((y2 - intercept)/slope)
    return np.array([x1, y1, x2, y2])


def average_slope_intercept(image, lines):
    left_fit = []
    right_fit = []
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        parameters = np.polyfit((x1, x2), (y1, y2), 1)
        slope = parameters[0]
        intercept = parameters[1]
        if slope < 0:
            left_fit.append((slope, intercept))
        else:
            right_fit.append((slope, intercept))
    left_fit_average = np.average(left_fit, axis=0)
    right_fit_average = np.average(right_fit, axis=0)
    left_line = make_coordinates(image, left_fit_average)
    right_line = make_coordinates(image, right_fit_average)
    return np.array([left_line, right_line])


def canny(image):
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image_blur = cv2.GaussianBlur(image_gray, (5, 5), 0)
    image_canny1 = cv2.Canny(image_blur, 50, 150)
    return image_canny1


def region_of_interest(image):
    height = image.shape[0]
    polygons = np.array([[200, height], [1100, height], [550, 250]])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, np.int32([polygons]), 255)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image


def display_lines(image, lines):
    line_image = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 10)
    return line_image

test_images_dir = join('road-image-master','test_image.jpg')
image1 = cv2.imread(test_images_dir)

cv2.imshow('image', image1)
cv2.waitKey(1)
# imageGray = cv2.COLOR_BGR2GRAY(image1)
# lane_image = np.copy(image1)
# canny_image = canny(lane_image)
# cropped_image = region_of_interest(canny_image)
# lines = cv2.HoughLinesP(cropped_image, 2, np.pi/180, 100, np.array([]), minLineLength = 42, maxLineGap = 5)
# average_lines = average_slope_intercept(lane_image, lines)
# line_image = display_lines(lane_image, average_lines)
# combo_image = cv2.addWeighted(lane_image, 0.8, line_image, 1, 1)

# cv2.imshow('result', combo_image)
# cv2.imshow('rest', image_canny)
# plt.imshow(image_canny)
# plt.show()

test_videos_dir = join('road-video-master','test2.mp4')
cap = cv2.VideoCapture(test_videos_dir)
print(cap.isOpened())
#while cap.isOpened():
#    print(1)
#    _, frame = cap.read()
#    canny_image = canny(frame)
#    cropped_image = region_of_interest(canny_image)
#    lines = cv2.HoughLinesP(cropped_image, 2, np.pi / 180, 100, np.array([]), minLineLength=42, maxLineGap=5)
#    average_lines = average_slope_intercept(frame, lines)
#    line_image = display_lines(frame, average_lines)
#    combo_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)
#    cv2.imshow('result', combo_image)
#    if cv2.waitKey(1) & 0xFF == ord('q'):
#        break

#cap1 = cv2.VideoCapture(0)
#
#while True:
#    print(cap1.isOpened())
#    ret, frame = cap1.read()
#    cv2.imshow('frame', frame)
#    if 0xFF == ord('q'):
#        break    

#cap1.release()
#cv2.destroyAllWindows()