


import json

import cv2
import numpy as np
from Image import Image

def load_config(filepath='config.json'):
    with open(filepath, 'r') as f:
        data = json.load(f)

    camera_matrix = np.array(data['camera_matrix'])
    dist_coeffs = np.array(data['dist_coeffs'])
    known_radius = data['KNOWN_RADIUS']
    focal_length = data['FOCAL_LENGTH']
    height_off_floor = data['HEIGHT_OFF_FLOOR']

    return camera_matrix, dist_coeffs, known_radius, focal_length, height_off_floor


mtx, dist, radius, fl, hof = load_config()
imgPros = Image(mtx, dist, radius, fl, hof)
print(f"Loaded Matrix:\n{mtx}")

img = cv2.imread('testImg/img1.jpg')
img = imgPros.convertToHSV(img)
img = imgPros.filterYellow(img)

cv2.imwrite('out.png', img)
