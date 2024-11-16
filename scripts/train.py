import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
import numpy as np
import time

# Parameters
num_classes = 8  # Number of scene classes
num_epochs = 25  # Number of training epochs
batch_size = 32  # Batch size
learning_rate = 0.001  # Learning rate

# Data Augmentation and Normalization
data_transforms = {
    "train": transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    ),
    "val": transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    ),
}

# Load Data
data_dir = "data"  # Directory containing train/val/test data
image_datasets = {
    x: datasets.ImageFolder(root=f"{data_dir}/{x}", transform=data_transforms[x])
    for x in ["train", "val"]
}
dataloaders = {
    x: DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=4)
    for x in ["train", "val"]
}
dataset_sizes = {x: len(image_datasets[x]) for x in ["train", "val"]}
class_names = image_datasets["train"].classes

# Load Pretrained ResNet and Modify for Scene Classification
model = models.resnet101(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Sequential(
    nn.Linear(num_ftrs, 512),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(512, num_classes),
    nn.Sigmoid(),  # For multi-label classification
)
model = model.to("cuda" if torch.cuda.is_available() else "cpu")

# Loss and Optimizer
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


# Training Function
def train_model(model, criterion, optimizer, num_epochs=25):
    best_model_wts = model.state_dict()
    best_f1 = 0.0

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print("-" * 10)

        # Each epoch has a training and validation phase
        for phase in ["train", "val"]:
            if phase == "train":
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_f1 = 0.0

            # Iterate over data
            for inputs, labels in dataloaders[phase]:
                inputs, labels = (
                    inputs.to(device),
                    labels.to(device).float(),
                )  # Convert labels to float for BCE loss

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward pass
                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    # Calculate F1 score
                    preds = (outputs > 0.5).float()  # Binary prediction
                    f1 = f1_score(
                        labels.cpu().numpy(), preds.cpu().numpy(), average="samples"
                    )

                    # Backward pass + Optimize only if in training phase
                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                # Track loss and F1 score
                running_loss += loss.item() * inputs.size(0)
                running_f1 += f1 * inputs.size(0)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_f1 = running_f1 / dataset_sizes[phase]

            print(
                f"{phase.capitalize()} Loss: {epoch_loss:.4f} F1 Score: {epoch_f1:.4f}"
            )

            # Save the model if it has the best F1 score so far
            if phase == "val" and epoch_f1 > best_f1:
                best_f1 = epoch_f1
                best_model_wts = model.state_dict()

    print(f"Best Val F1 Score: {best_f1:.4f}")

    # Load best model weights
    model.load_state_dict(best_model_wts)
    return model


# Train the model
device = "cuda" if torch.cuda.is_available() else "cpu"
model = train_model(model, criterion, optimizer, num_epochs=num_epochs)

# Save the trained model
torch.save(model.state_dict(), "resnet101_scene_classification.pth")
