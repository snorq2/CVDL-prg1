import numpy as np
from skimage import io
from skimage.color import rgb2gray
import skimage.feature as ft
from matplotlib import pyplot as plt

img1 = io.imread('SIFT1_img.jpg')
img2 = io.imread('SIFT2_img.jpg')

img1_gray = rgb2gray(img1)
img2_gray = rgb2gray(img2)

def get_SIFT(img):
    descriptor_extractor = ft.SIFT()
    descriptor_extractor.detect_and_extract(img)
    return descriptor_extractor.keypoints, descriptor_extractor.descriptors

def L2(desc_1, desc_2):
    pretotal = (desc_1 - desc_2) ** 2
    sumtotal = np.sum(pretotal)
    return np.sqrt(sumtotal)

def descriptor_compare(desc_1, desc_2):
    matches = np.full((len(desc_1), 2), np.ma.minimum_fill_value(np.float64()))
    for i in range(0, len(desc_1)-1):
        for j in range(0, len(desc_2)-1):
            l2 = L2(desc_1[i], desc_2[j])
            if l2 < matches[i, 1]:
                matches[i, 0] = j
                matches[i, 1] = l2
    return matches

keys_1, desc_1 = get_SIFT(img1_gray)
keys_2, desc_2 = get_SIFT(img2_gray)

matches = descriptor_compare(desc_1, desc_2)
print(matches)
# io.imshow(img1)
# plt.scatter(keys_1[:,1], keys_1[:,0], c='r', s=1)
# plt.show()
# input("Press enter to continue...")

# io.imshow(img2)
# plt.scatter(keys_2[:,1], keys_2[:,0], c='r', s=1)
# plt.show()
