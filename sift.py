import numpy as np
from skimage import io
from skimage.color import rgb2gray
import skimage.feature as ft
from matplotlib import pyplot as plt

img1 = io.imread('SIFT1_img.jpg')
img2 = io.imread('SIFT2_img.jpg')

img1_gray = rgb2gray(img1)
img2_gray = rgb2gray(img2)

def get_keypoints(img):
    descriptor_extractor = ft.SIFT()
    descriptor_extractor.detect_and_extract(img)
    return descriptor_extractor.keypoints

keys_1 = get_keypoints(img1_gray)
keys_2 = get_keypoints(img2_gray)
io.imshow(img1)
plt.scatter(keys_1[:,1], keys_1[:,0], c='r', s=1)
plt.show()
input("Press enter to continue...")

io.imshow(img2)
plt.scatter(keys_2[:,1], keys_2[:,0], c='r', s=1)
plt.show()

