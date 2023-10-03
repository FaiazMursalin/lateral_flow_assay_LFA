# importing relevant packages
import cv2
import os
import matplotlib.pyplot as plt
import numpy as np

image_directory = 'D:\MSU\semester-1\Deep Learning\lateral_flow_assay_LFA\images'


# Load an image from the directory
image_path = os.path.join(image_directory,'Picture1.png')
image = cv2.imread(image_path)

# Display the original image using matplotlib
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title('Original Image')
plt.show()

# Convert the image to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Display the grayscale image
plt.imshow(gray_image, cmap='gray')
plt.title('Grayscale Image')
plt.show()

# Define the lower and upper bounds for red color in BGR
lower_red = np.array([100, 0, 0])
upper_red = np.array([255,100, 100])
# Create a mask to identify red regions
red_mask = cv2.inRange(image, lower_red, upper_red)

# Apply the mask to the grayscale image
red_lines_gray = cv2.bitwise_and(gray_image, gray_image, mask=red_mask)


# Detect lines in the grayscale image using the Hough Line Transform
lines = cv2.HoughLinesP(red_lines_gray, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=10)

# Draw the detected lines on a copy of the original image
image_with_lines = image.copy()

if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(image_with_lines, (x1, y1), (x2, y2), (0, 255, 0), 10)  # Draw lines in green

# Display the image with marked red lines
plt.imshow(cv2.cvtColor(image_with_lines, cv2.COLOR_BGR2RGB))
plt.title('Image with Red Horizontal Lines Marked')
plt.axis('off')
plt.show()
