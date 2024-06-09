import os
import cv2
import numpy as np


def count_birds(image):
    # Converting into gray scale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Applying Gaussian Blur to reduce noise and improve edge detection
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Adaptive thresholding to better handle varying lighting conditions
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)

    # Morphological operations to close gaps and enhance contours
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # Finding contours
    contours, _ = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Counting birds
    bird_count = len(contours)

    return bird_count


# Find and process all images
def process_images_in_folder(folder_path):
    for filename in os.listdir(folder_path):
        if filename.endswith(".jpg"):
            image_path = os.path.join(folder_path, filename)
            image = cv2.imread(image_path)
            bird_count = count_birds(image)
            print(f"Obraz: {filename}, Liczba ptaków: {bird_count}")


# Path to the folder
folder_path = "bird_miniatures"

process_images_in_folder(folder_path)
