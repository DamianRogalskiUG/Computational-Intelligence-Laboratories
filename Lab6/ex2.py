import cv2
import numpy as np
import matplotlib.pyplot as plt


def convert_to_gray_avg(img):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return gray_img

def convert_to_gray_weighted(img):
    gray_img = np.dot(img[...,:3], [0.299, 0.587, 0.114])
    return gray_img.astype(np.uint8)


# Load image
img_color = cv2.imread('ex2_image.jpg')

# Convert via avg
gray_avg = convert_to_gray_avg(img_color)

# Convert via sum
gray_weighted = convert_to_gray_weighted(img_color)

# Show images
plt.figure(figsize=(10, 5))

plt.subplot(1, 3, 1)
plt.imshow(cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB))
plt.title('Coloured Image')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(gray_avg, cmap='gray')
plt.title('Avg Method')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(gray_weighted, cmap='gray')
plt.title('Sum Method')
plt.axis('off')

plt.show()
