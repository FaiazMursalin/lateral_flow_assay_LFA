# importing relevant packages
import cv2
import os
import matplotlib.pyplot as plt

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
