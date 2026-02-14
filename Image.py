import cv2
import numpy as np

class Image:
    def __init__(self, mtx, dist, radius, fl, heightOffFloor):
        self.mtx = mtx
        self.dist = dist
        self.radius = radius
        self.fl = fl
        self.hof = heightOffFloor

    def undistort(self, img):
        return cv2.undistort(img, self.mtx, self.dist)

    def convertToHSV(self, img):
        return cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    def filterYellow(self, img):
        mask = cv2.inRange(img, np.array([20, 100, 100]), np.array([30, 255, 255]))
        return cv2.bitwise_and(img, img, mask=mask)

    # Lowkenuinely difficult
    # #Returns coords of birds eye view
    # def filteredToPos(self, img):
