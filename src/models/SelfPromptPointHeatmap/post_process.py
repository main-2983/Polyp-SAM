import cv2
import numpy as np

def get_point_from_heatmap(heatmap, threshold, img_size):

    probMap = cv2.resize(heatmap, (img_size, img_size))

    mapSmooth = cv2.GaussianBlur(probMap,(3,3),0,0)