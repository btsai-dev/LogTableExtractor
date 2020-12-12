import cv2
import numpy as np


def detect_box(img):
    kernel_len = np.array(img).shape[1]

    kernel_vert = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_len))
    kernel_hori = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_len, 1))

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    img_vert = cv2.erode(img, kernel_vert, iterations=3)
    img_vert = cv2.dilate(img_vert, kernel_vert, iterations=3)

    img_hori = cv2.erode(img, kernel_hori, iterations=3)
    img_hori = cv2.dilate(img_hori, kernel_hori, iterations=3)