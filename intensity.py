import cv2
import matplotlib.pyplot as plt
import numpy as np
from rembg import remove

from segmentation_ROI import UNET
from utils import load_checkpoint
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
import torchvision
from torchvision.transforms import transforms


def process_lfa_strip(image_path):
    # Read the cropped LFA strip image
    lfa_image = cv2.imread(image_path)
    # Display the image with contours
    lfa_image = cv2.resize(lfa_image, (1600, 1200))
    gray_image = cv2.cvtColor(lfa_image, cv2.COLOR_BGR2GRAY)
    contours, _ = cv2.findContours(gray_image, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv2.contourArea)

    # cv2.drawContours(gray_image, [largest_contour], -1, (0, 255, 0), thickness=1)  # Change color and thickness as needed
    # cv2.imshow("Cropped ROI", gray_image)
    # cv2.waitKey(0)

    # Apply Sobel filter to emphasize edges
    sobel_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
    # Combine the x and y gradients to get the overall gradient magnitude
    gradient_magnitude = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
    # Normalize the gradient magnitude to the range [0, 255]
    normalized_gradient = cv2.normalize(gradient_magnitude, None, 0, 255, cv2.NORM_MINMAX)
    # Convert the result to uint8
    normalized_gradient = np.uint8(normalized_gradient)
    # Draw contours on the original image
    # cv2.drawContours(lfa_image, contours, -1, (0, 255, 0), thickness=2)  # Change color and thickness as needed
    # cv2.imshow("Cropped ROI", lfa_image)
    # cv2.imwrite("Normalized.jpg", normalized_gradient)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    bounding_box = cv2.boundingRect(largest_contour)
    # print("bnd box", bounding_box)
    x, y, w, h = bounding_box
    intensities = []
    for dy in range(y, y + h):
        line = normalized_gradient[
               dy: dy + 1,
               x: x + w
               ]
        intensities.append(np.mean(line))
    # plt.plot(intensities[5:-5])
    # plt.show()
    #
    # new_img = lfa_image[bounding_box[1]:bounding_box[1] + bounding_box[3],
    #           bounding_box[0]:bounding_box[0] + bounding_box[2]]
    # plt.imshow(new_img)
    # Plot intensities
    plt.figure(figsize=(8, 5))
    plt.subplot(211)
    plt.plot(intensities[20:-20])
    plt.title('Intensity Plot along with LFA Strip')

    # Display the cropped image
    new_img = lfa_image[bounding_box[1]:bounding_box[1] + bounding_box[3],
                  bounding_box[0]:bounding_box[0] + bounding_box[2]]

    plt.subplot(212)
    plt.imshow(cv2.cvtColor(new_img, cv2.COLOR_BGR2RGB))  # Convert BGR to RGB for Matplotlib
    plt.title('Cropped Image')

    plt.tight_layout()
    plt.show()


def crop(original_image, binary_mask):
    # Find contours in the binary mask
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        largest_contour = max(contours, key=cv2.contourArea)

        # Remove Outliers
        out = np.zeros(binary_mask.shape, np.uint8)
        cv2.drawContours(out, [largest_contour], -1, 255, cv2.FILLED)
        binary_mask = cv2.bitwise_and(binary_mask, out)

        # # Blur the image for better edge detection
        # blurred = cv2.GaussianBlur(binary_mask, (3, 3), 0)
        # # Canny Edge Detection
        # edged = cv2.Canny(image=blurred, threshold1=10, threshold2=90)
        #
        # # cv2.imshow("Canny edge detection", edged)
        # # cv2.waitKey(0)
        #
        # # find the contours in the edged image
        # contours, hierarchy = cv2.findContours(edged, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        # # image_copy = lfa_image.copy()
        # # draw the contours on a copy of the original image
        # # contours.remove(max(contours, key=cv2.contourArea))
        # contours = contours[:13]
        # cv2.drawContours(binary_mask, contours, -1, (0, 255, 0), 2)
        # print(len(contours), "objects were found in this image.")
        #
        # cv2.imshow("contours", cv2.resize(binary_mask, (1600, 1200)))
        # cv2.waitKey(0)


        cv2.drawContours(binary_mask, contours, 0, (255, 255, 255), thickness=cv2.FILLED)
        original_image_with_alpha = cv2.merge((original_image, binary_mask))
        roi_image = original_image_with_alpha.copy()
        roi_image = remove(roi_image)
        # roi_image[binary_mask == 0] = [0, 0, 0, 0]
        # processed_image_path = 'data/'
        cv2.imwrite("./saved_images/test/cropped_image.png", roi_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("No contours found in the binary mask.")


def single_image_inference(image_pth, model_pth, device, folder="./saved_images/test"):
    checkpoint = torch.load(model_pth, map_location=torch.device('cpu'))
    model = UNET(in_channels=3, out_channels=1).to(device)
    load_checkpoint(checkpoint, model)

    transform = A.Compose(
        [
            A.Resize(height=512, width=512),
            ToTensorV2()
        ]
    )
    # Image.open(image_pth).show()

    img = transform(image=np.array(Image.open(image_pth).convert("RGB"), dtype=np.float32))["image"].to(device)
    img = img.unsqueeze(0)
    with torch.no_grad():
        preds = torch.sigmoid(model(img))
        # img = img.squeeze(0).cpu().detach()
        preds = (preds > 0.5).float()
    torchvision.utils.save_image(
        transforms.Resize((3000, 4000))(preds), f"{folder}/pred.png"
    )
    # torchvision.utils.save_image(img.unsqueeze(1), f"{folder}/actual.png")
    return f"{folder}/pred.png"


if __name__ == "__main__":
    # # Load the original image and the binary mask
    # original_image = cv2.imread("data/test_images/image_1.jpg")
    # # utilize model here to get the predicted mask
    # single_image_inference()
    # binary_mask = cv2.imread("data/tests_mask/image_1_mask.png", cv2.IMREAD_GRAYSCALE)
    # crop(original_image, binary_mask)
    # # process lfa strip for graph   rotated_image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    # process_lfa_strip('cropped_image.png')

    #########################################
    original_image_path = "data/test_images/image_1.jpg"
    model_path = 'models/my_checkpoint.pth.tar'
    device = "cpu"
    pred_mask_path = single_image_inference(original_image_path, model_path, device)
    original_image = cv2.imread(original_image_path)
    pred_binary_mask = cv2.imread(pred_mask_path, cv2.IMREAD_GRAYSCALE)
    pred_binary_mask_rotated = cv2.rotate(pred_binary_mask, cv2.ROTATE_90_CLOCKWISE)
    crop(original_image, pred_binary_mask_rotated)
    process_lfa_strip('saved_images/test/cropped_image.png')
    # cropped_image = cv2.imread("saved_images/test/cropped_image.png")
    # plt.imshow(cropped_image)
    # plt.title('cropped image')
    # plt.show()
