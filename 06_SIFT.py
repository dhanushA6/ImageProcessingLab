import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('dog1.jpeg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

sift = cv2.SIFT_create()

keypoints, descriptors = sift.detectAndCompute(gray, None)

sift_image = cv2.drawKeypoints(
    gray, keypoints, None,
    flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
)

plt.figure(figsize=(10,6))
plt.imshow(sift_image, cmap='gray')
plt.title(f"Detected SIFT Keypoints: {len(keypoints)}")
plt.axis('off')
plt.show()