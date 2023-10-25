import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm  # for progress
import torch.nn as nn
import torch.optim as optim
from segmentation_ROI import UNET

from utils import (load_checkpoint, save_checkpoint, get_loaders, check_accuracy, save_predictions_as_image, )

# hyperparameters
LEARNING_RATE = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32
NUM_EPOCHS = 100
NUM_WORKERS = 2
IMAGE_HEIGHT = 160
IMAGE_WIDTH = 240
PIN_MEMORY = True
LOAD_MODEL = False  # for further epochs when i have already saved training then we can turn it to true
TRAIN_IMAGE_DIRECTORY = ""
TRAIN_MASK_DIRECTORY = ""
VAL_IMAGE_DIRECTORY = ""
VAL_MASK_DIRECTORY = ""


def train_fn(loader, model, optimizer, loss_fn, scaler):
    loop = tqdm(loader)
    for batch_index, (data, targets) in enumerate(loop):
        data = data.to(device=DEVICE)
        targets = targets.float().unsqueeze(1).to(device=DEVICE)

        # forward
        with torch.cuda.amp.autocast():
            predictions = model(data)
            loss = loss_fn(predictions, targets)

        # backward propagation
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # update tqdm loop
        loop.set_postfix(loss=loss.item())


def main():
    train_transform = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Rotate(limit=35, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ]
    )
    val_transform = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ]
    )
    model = UNET(in_channels=3, out_channels=1).to(DEVICE)  # as the mask is binary so i used out as 1
    loss_fn = nn.BCEWithLogitsLoss()  # cross entropy without sigmoid in the outer layer
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_loader, val_loader = get_loaders(
        TRAIN_IMAGE_DIRECTORY,
        TRAIN_MASK_DIRECTORY,
        VAL_IMAGE_DIRECTORY,
        VAL_MASK_DIRECTORY,
        BATCH_SIZE,
        train_transform,
        val_transform,
        NUM_WORKERS,
        PIN_MEMORY,
    )

    if LOAD_MODEL:
        load_checkpoint(torch.load("my_checkpoint.pth.tar"), model)
    check_accuracy(val_loader, model, device=DEVICE)
    scaler = torch.cuda.amp.GradScaler()
    for epoch in range(NUM_EPOCHS):
        train_fn(train_loader, model, optimizer, loss_fn, scaler)

        # save model
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        save_checkpoint(checkpoint)
        # check accuracy
        check_accuracy(val_loader, model, device=DEVICE)
        # print some examples to a fold
        save_predictions_as_image(val_loader, model, folder="saved_images/", device=DEVICE)


if __name__ == "__main__":
    main()
