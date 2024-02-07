import re
import matplotlib.pyplot as plt

# Initialize lists to store extracted values
train_losses = []
train_accuracies = []
val_losses = []
val_accuracies = []

with open('result.txt', 'r') as file:
    lines = file.readlines()

for line in lines:
    if line.startswith('Train Losses'):
        train_losses = list(map(float, re.findall(r"[-+]?\d*\.\d+|\d+", line)))
    elif line.startswith('Train Accuracies'):
        train_accuracies = list(map(float, re.findall(r"[-+]?\d*\.\d+|\d+", line)))
    elif line.startswith('Val Losses'):
        val_losses = list(map(float, re.findall(r"[-+]?\d*\.\d+|\d+", line)))
    elif line.startswith('Val Accuracies'):
        val_accuracies = list(map(float, re.findall(r"[-+]?\d*\.\d+|\d+", line)))

epochs = list(range(1, len(train_losses) + 1))  # Assuming lengths are consistent across all metrics

# Plotting train losses and validation losses
plt.figure(figsize=(10, 6))
plt.plot(epochs, train_losses, label='Train Losses')
plt.plot(epochs, val_losses, label='Validation Losses')
plt.title('Train and Validation Losses across Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()

# Plotting train accuracies and validation accuracies
plt.figure(figsize=(10, 6))
plt.plot(epochs, train_accuracies, label='Train Accuracies')
plt.plot(epochs, val_accuracies, label='Validation Accuracies')
plt.title('Train and Validation Accuracies across Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.show()