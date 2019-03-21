#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import sys
import os


def faceRecognition(cascade, test_image, scaleFactor=1.1):
    image_copy = test_image.copy()
    # Convert the test image to gray scale as opencv face detector
    # expects gray images
    gray_image = cv2.cvtColor(image_copy, cv2.COLOR_BGR2GRAY)
    # Applying the haar classifier to detect faces
    faces_rect = cascade.detectMultiScale(
        gray_image, scaleFactor=scaleFactor, minNeighbors=5)
    for (x, y, w, h) in faces_rect:
        cv2.rectangle(image_copy, (x, y), (x+w, y+h), (0, 255, 0), 2)

    return image_copy


def imageExists(imagePath):
    """
    Check if the image in the path exists.
    """
    exists = os.path.isfile(imagePath)
    if exists:
        return True
    return False


def help(execName):
    print("FaceID Microprocessor in Raspberry Pi\n")
    print("\t python " + execName + " <path_file_name.[jpg|png]>")
    print("\t python " + execName + " -h")
    print("\nThis program needs conda env to run Open CV.")


# Start
if __name__ == '__main__':
    """
    This program identify face in image.
        python faceid <path_file_name.[jpg|png]> | -h
    Returns:
        1 - Bad Input
        2 - Error. None faces.
        3 - Error. The input not exists.
        0 - If there were a face in the image
    """
    # Get args
    if len(sys.argv) == 1:
        print("Argumentos faltantes. \n")
        help(sys.argv[0])
        sys.exit(1)

    arg1 = sys.argv[1]
    if arg1 == "-h":
        help(sys.argv[0])
        sys.exit(1)
    
    if imageExists(arg1) is True:
        org_image = cv2.imread(arg1)
        filter = cv2.CascadeClassifier(
            './haarcascade/haarcascade_frontalface_default.xml')
        face_detection = faceRecognition(filter, org_image)
        cv2.imwrite('changed.jpg',face_detection)
        sys.exit(0)
    else:
        print("Archivo no existente.")
        print("Asegurate de que tienes el path correcto.")
        sys.exit(1)
