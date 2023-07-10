import cv2
import numpy as np
from matplotlib import pyplot as plt

img_1 = cv2.imread('SIFT1_img.jpg')
gray1 = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
img_2 = cv2.imread('SIFT2_img.jpg')
gray2 = cv2.cvtColor(img_2, cv2.COLOR_BGR2GRAY)

sift = cv2.SIFT_create()
kp1, des1 = sift.detectAndCompute(gray1, None)
kp2, des2 = sift.detectAndCompute(gray2, None)

bf = cv2.BFMatcher(cv2.NORM_L2)

matches = bf.match(des1, des2)

matches = sorted(matches, key = lambda x:x.distance)
img_match = cv2.drawMatches(img_1, kp1, img_2, kp2, matches[:700], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

cv2.imwrite('SIFT_matches.jpg', img_match)

# imgkp = 0
# imgkp = cv2.drawKeypoints(img_1, kp1, imgkp)
# cv2.imwrite('SIFT2_img_kp.jpg', imgkp)
# cv2.imshow('kp', imgkp)
# cv2.waitKey()