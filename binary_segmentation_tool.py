import cv2
import os
import numpy as np

# Directory paths
input_directory = '/home/faiaz/Documents/DLProject_LFA/all_images'  # Replace with your input directory path
output_directory = '/home/faiaz/Documents/DLProject_LFA/final_images_mask'  # Replace with your output directory path


# Create the output directory if it doesn't exist
os.makedirs(output_directory, exist_ok=True)

# Function to handle mouse events
def draw_polygon(event, x, y, flags, param):
    global points

    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        if len(points) > 1:
            cv2.line(img, points[-2], points[-1], (0, 255, 0), 1)
            cv2.imshow('Image for Annotation', img)

    elif event == cv2.EVENT_RBUTTONDOWN:
        if len(points) > 1:
            cv2.line(img, points[-1], points[0], (0, 255, 0), 1)
            cv2.imshow('Image for Annotation', img)

            # Create mask based on the polygon
            mask = np.zeros_like(img[:, :, 0])
            polygon = np.array([points], np.int32)
            cv2.fillPoly(mask, polygon, 255)

            # Save black and white mask image of the same size as the original image
            mask_image = np.zeros_like(img)
            mask_image[mask == 255] = 255  # Set the inside of the polygon to white

            # Save the masked image
            filename = os.path.basename(image_path)
            mask_output_path = os.path.join(output_directory, f"{os.path.splitext(filename)[0]}_mask.png")
            cv2.imwrite(mask_output_path, mask_image)

            # Reset points for a new annotation
            points = []
            cv2.imshow('Image for Annotation', img)


# Process each image in the input directory
for filename in os.listdir(input_directory):
    if filename.endswith('.jpg') or filename.endswith('.png'):  # Process only image files
        image_path = os.path.join(input_directory, filename)
        img = cv2.imread(image_path)

        cv2.namedWindow('Image for Annotation', cv2.WINDOW_NORMAL)
        cv2.setMouseCallback('Image for Annotation', draw_polygon)
        points = []

        while True:
            cv2.imshow('Image for Annotation', img)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # Press Esc to move to the next image
                break

cv2.destroyAllWindows()
