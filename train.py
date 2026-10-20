import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp

from dataset import SegmentationDataset
from metrics import evaluate_all    # ✅ NEW METRICS

device = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------------
# Load datasets
# -------------------------

train_dataset = SegmentationDataset(
    "dataset/train/images",
    "dataset/train/masks"
)

val_dataset = SegmentationDataset(
    "dataset/val/images",
    "dataset/val/masks"
)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

# -------------------------
# Model
# -------------------------

model = smp.Unet(
    encoder_name="resnet34",
    in_channels=3,
    classes=1
).to(device)

loss_fn = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

epochs = 15

best_val_accuracy = 0
best_epoch = 0

# -------------------------
# Training Loop
# -------------------------

for epoch in range(epochs):

    model.train()

    train_loss = 0
    train_accuracy = 0

    for images, masks in train_loader:

        images = images.to(device)
        masks = masks.to(device)

        preds = model(images)

        loss = loss_fn(preds, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        preds_bin = torch.sigmoid(preds)
        preds_bin = (preds_bin > 0.5).float()

        correct = (preds_bin == masks).float().sum()
        accuracy = correct / torch.numel(preds_bin)

        train_accuracy += accuracy.item()

    train_loss /= len(train_loader)
    train_accuracy /= len(train_loader)

    # -------------------------
    # Validation
    # -------------------------

    model.eval()

    val_accuracy = 0
    val_dice = 0
    val_iou = 0

    with torch.no_grad():

        for images, masks in val_loader:

            images = images.to(device)
            masks = masks.to(device)

            preds = model(images)

            preds_bin = torch.sigmoid(preds)
            preds_bin = (preds_bin > 0.5).float()

            # Accuracy
            correct = (preds_bin == masks).float().sum()
            accuracy = correct / torch.numel(preds_bin)
            val_accuracy += accuracy.item()

            # ✅ Convert to numpy for metrics
            pred_np = preds_bin.cpu().numpy()
            mask_np = masks.cpu().numpy()

            metrics = evaluate_all(pred_np, mask_np)

            val_dice += metrics["Dice"]
            val_iou += metrics["IoU"]

    val_accuracy /= len(val_loader)
    val_dice /= len(val_loader)
    val_iou /= len(val_loader)

    # -------------------------
    # Print results
    # -------------------------

    print(f"Epoch {epoch+1}/{epochs}")
    print(f"Train Loss: {train_loss:.4f}")
    print(f"Train Accuracy: {train_accuracy:.4f}")
    print(f"Validation Accuracy: {val_accuracy:.4f}")
    print(f"Validation Dice: {val_dice:.4f}")
    print(f"Validation IoU: {val_iou:.4f}")
    print("--------------------------------------------------")

    # -------------------------
    # Save best model
    # -------------------------

    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        best_epoch = epoch + 1
        torch.save(model.state_dict(), "best_model.pth")

# -------------------------
# Training Complete
# -------------------------

print("Training Finished")
print(f"Best Epoch: {best_epoch}")
print(f"Best Validation Accuracy: {best_val_accuracy:.4f}")