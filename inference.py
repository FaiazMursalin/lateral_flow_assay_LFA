import numpy as np
import torch
import matplotlib.pyplot as plt
import torchvision
from albumentations.pytorch import ToTensorV2
from PIL import Image
import albumentations as A
from torchvision.transforms import transforms

from dataset import LFADataset
from lateral_flow_assay_LFA.utils import load_checkpoint
from segmentation_ROI import UNET


def pred_show_image_grid(data_path, model_pth, device):
    model = UNET(in_channels=3, out_channels=1).to(device)
    model.load_state_dict(torch.load(model_pth, map_location=torch.device(device)))
    image_dataset = LFADataset(data_path)
    images = []
    orig_masks = []
    pred_masks = []

    for img, orig_mask in image_dataset:
        img = img.float().to(device)
        img = img.unsqueeze(0)

        pred_mask = model(img)

        img = img.squeeze(0).cpu().detach()
        img = img.permute(1, 2, 0)

        pred_mask = pred_mask.squeeze(0).cpu().detach()
        pred_mask = pred_mask.permute(1, 2, 0)
        pred_mask[pred_mask < 0] = 0
        pred_mask[pred_mask > 0] = 1

        orig_mask = orig_mask.cpu().detach()
        orig_mask = orig_mask.permute(1, 2, 0)

        images.append(img)
        orig_masks.append(orig_mask)
        pred_masks.append(pred_mask)

    images.extend(orig_masks)
    images.extend(pred_masks)
    fig = plt.figure()
    for i in range(1, 3 * len(image_dataset) + 1):
        fig.add_subplot(3, len(image_dataset), i)
        plt.imshow(images[i - 1], cmap="gray")
    plt.show()


def single_image_inference(image_pth, model_pth, device, folder="./saved_images/test"):
    model = UNET(in_channels=3, out_channels=1).to(device)
    load_checkpoint(torch.load(model_pth), model)

    transform = A.Compose(
        [
            A.Resize(height=512, width=512),
            ToTensorV2()
        ]
    )
    Image.open(image_pth).show()

    img = transform(image=np.array(Image.open(image_pth).convert("RGB"), dtype=np.float32))["image"].to(device)
    img = img.unsqueeze(0)
    with torch.no_grad():
        preds = torch.sigmoid(model(img))
        img = img.squeeze(0).cpu().detach()
        preds = (preds > 0.5).float()
    torchvision.utils.save_image(
        transforms.Resize((3000, 4000))(preds), f"{folder}/pred.png"
    )
    torchvision.utils.save_image(img.unsqueeze(1), f"{folder}/actual.png")


if __name__ == "__main__":
    SINGLE_IMG_PATH = "data/train_images/image_481.jpg"
    DATA_PATH = "./data"
    MODEL_PATH = "./models/unet.pth.tar"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # pred_show_image_grid(DATA_PATH, MODEL_PATH, device)
    single_image_inference(SINGLE_IMG_PATH, MODEL_PATH, device)
