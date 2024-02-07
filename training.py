import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm  # for progress
import torch.nn as nn
import torch.optim as optim
from segmentation_ROI import UNET

from utils import (load_checkpoint, save_checkpoint, get_loaders, check_accuracy, save_predictions_as_image, )

# hyperparameters
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 8
NUM_EPOCHS = 2000
NUM_WORKERS = 2
IMAGE_HEIGHT = 512
IMAGE_WIDTH = 512
PIN_MEMORY = True
LOAD_MODEL = False  # for further epochs when i have already saved training then we can turn it to true


def train_fn(loader, model, optimizer, loss_fn, scaler):
    train_loss = 0.0
    num_correct = 0
    num_pixels = 0
    current_lr = optimizer.param_groups[0]["lr"]
    loop = tqdm(loader)
    for batch_index, (data, targets) in enumerate(loop):
        torch.cuda.empty_cache()
        data = data.to(device=DEVICE)
        targets = targets.float().unsqueeze(1).to(device=DEVICE)

        # forward
        with torch.cuda.amp.autocast():
            predictions = torch.sigmoid(model(data))
            loss = loss_fn(predictions, targets)

            preds = predictions
            preds = (preds > 0.5).float()
            num_correct += (preds == targets).sum()
            num_pixels += torch.numel(preds)

        # backward propagation
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # update tqdm loop
        loop.set_postfix(
            loss=loss.item(),
            lr=current_lr
        )
        train_loss += loss.item()

    accuracy = float(f"{num_correct / num_pixels * 100:.2f}")

    return train_loss / len(loader), accuracy


def main():
    train_transform = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Rotate(limit=35, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            A.RandomCrop(height=IMAGE_HEIGHT, width=IMAGE_WIDTH, p=0.7),
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
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-6)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=3, factor=0.1, verbose=True
    )

    train_loader, val_loader = get_loaders(
        BATCH_SIZE,
        train_transform,
        val_transform,
        NUM_WORKERS,
        PIN_MEMORY,
    )

    accuracy = 0.0

    if LOAD_MODEL:
        load_checkpoint(torch.load("models/my_checkpoint.pth.tar"), model)
        _, val_accuracy = check_accuracy(val_loader, model, device=DEVICE)
        accuracy = max(val_accuracy, accuracy)

    scaler = torch.cuda.amp.GradScaler()

    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []
    for epoch in range(NUM_EPOCHS):
        train_loss, train_accuracy = train_fn(train_loader, model, optimizer, loss_fn, scaler)
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        print(f"Epoch {epoch}: Training Loss: {train_loss} and Accuracy {train_accuracy}")

        # check accuracy
        val_loss, val_accuracy = check_accuracy(val_loader, model, device=DEVICE, loss_fn=loss_fn)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)
        print(f"Epoch {epoch}: Validation Loss: {val_loss} and Accuracy {val_accuracy}")
        scheduler.step(val_loss)

        if val_accuracy > accuracy:
            accuracy = val_accuracy

            # save model
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            save_checkpoint(checkpoint)

        save_predictions_as_image(val_loader, model, folder="./saved_images/", device=DEVICE)

        print(f"Epoch {epoch + 1}")
        print("Train Losses: ", train_losses)
        print("Train Accuracies: ", train_accuracies)
        print("Val Losses: ", val_losses)
        print("Val Accuracies: ", val_accuracies)


if __name__ == "__main__":
    main()
