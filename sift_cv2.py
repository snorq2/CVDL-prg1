import cv2

# Load and convert the two images
img_1 = cv2.imread('SIFT1_img.jpg')
gray1 = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
img_2 = cv2.imread('SIFT2_img.jpg')
gray2 = cv2.cvtColor(img_2, cv2.COLOR_BGR2GRAY)

# Create and execute the SIFT filters
sift = cv2.SIFT_create()
kp1, des1 = sift.detectAndCompute(gray1, None)
kp2, des2 = sift.detectAndCompute(gray2, None)

# Set up the keypoint matches and sort them lowest L2 distance to highest
bf = cv2.BFMatcher(cv2.NORM_L2)
matches = bf.match(des1, des2)
matches = sorted(matches, key = lambda x:x.distance)

# Match the keypoints by selecting the top 10% of the sorted array rows
img_match = cv2.drawMatches(img_1, kp1, img_2, kp2, matches[:int(len(matches)/10)], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
cv2.imwrite('SIFT_matches.jpg', img_match)

# Overlay the keypoints onto the original images and save
imgkp = 0
imgkp = cv2.drawKeypoints(img_1, kp1, imgkp)
cv2.imwrite('SIFT1_img_kp.jpg', imgkp)
imgkp = cv2.drawKeypoints(img_2, kp2, imgkp)
cv2.imwrite('SIFT2_img_kp.jpg', imgkp)