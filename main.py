import numpy as np
import cv2
import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt


def convertToRGB(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Import haar cascade to detect face
haar_cascade_face = cv2.CascadeClassifier('./haarcascade/haarcascade_frontalface_default.xml')
# print(haar_cascade_face)

# Load the test image
test_image = cv2.imread('./baby1.jpg')

# Convert grayscale
test_image_gray = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)

# Display gray scale image
plt.imshow(test_image_gray, cmap='gray')

faces_rects = haar_cascade_face.detectMultiScale(test_image_gray, scaleFactor = 1.2, minNeighbors = 5)

# Let us print the no. of faces found
print('Faces found: ', len(faces_rects))

for (x,y,w,h) in faces_rects:
     cv2.rectangle(test_image, (x, y), (x+w, y+h), (0, 255, 0), 2)

plt.imshow(convertToRGB(test_image))

plt.show()


