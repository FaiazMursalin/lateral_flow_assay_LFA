import torch
import torchvision
from dataset import LFADataset
from torch.utils.data import DataLoader, random_split


def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)


def load_checkpoint(checkpoint, model):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])


def get_loaders(
        batch_size,
        train_transform,
        val_transform,
        num_workers=4,
        pin_memory=True,
):
    train_ds = LFADataset(
        root_path="./data/",
        transform=train_transform,
    )

    generator = torch.Generator().manual_seed(42)
    train_ds, val_ds = random_split(train_ds, [0.8, 0.2], generator=generator)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
    )
    return train_loader, val_loader


def check_accuracy(loader, model, device="cpu", loss_fn=None):
    loss = 0.0
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device).unsqueeze(1)  # because label doesnot have channel
            preds = torch.sigmoid(model(x))
            if loss_fn:
                loss += loss_fn(preds, y)

            preds = (preds > 0.5).float()
            num_correct += (preds == y).sum()
            num_pixels += torch.numel(preds)
            dice_score += (2 * (preds * y).sum()) / ((preds + y).sum() + 1e-8)

    model.train()

    loss /= len(loader)
    accuracy = float(f"{num_correct / num_pixels * 100:.2f}")

    print(f"Got Loss {loss}")
    print(f"Got {num_correct}/{num_pixels} with accuracy {accuracy}")
    print(f"Dice score: {dice_score / len(loader)}")

    return loss, accuracy


def save_predictions_as_image(loader, model, folder, device="cpu"):
    model.eval()
    for index, (x, y) in enumerate(loader):
        if index == 5:
            break
        x = x.to(device=device)
        with torch.no_grad():
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
        torchvision.utils.save_image(
            preds, f"{folder}/pred_{index}.png"
        )
        torchvision.utils.save_image(y.unsqueeze(1), f"{folder}/actual_{index}.png")
    model.train()
