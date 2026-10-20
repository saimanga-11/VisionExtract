import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp

from dataset import SegmentationDataset
from metrics import evaluate_all    # ✅ NEW METRICS
from prepare_data import DATASET_DIR, prepare_data

device = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------------
# Load datasets
# -------------------------

splits = prepare_data()
train_image_dir = DATASET_DIR / "DUTS-TR" / "DUTS-TR-Image"
train_mask_dir = DATASET_DIR / "DUTS-TR" / "DUTS-TR-Mask"
test_image_dir = DATASET_DIR / "DUTS-TE" / "DUTS-TE-Image"
test_mask_dir = DATASET_DIR / "DUTS-TE" / "DUTS-TE-Mask"

train_dataset = SegmentationDataset(
    str(train_image_dir),
    str(train_mask_dir),
    image_names=splits["train"]
)
val_dataset = SegmentationDataset(
    str(train_image_dir),
    str(train_mask_dir),
    image_names=splits["val"]
)
test_dataset = SegmentationDataset(
    str(test_image_dir),
    str(test_mask_dir),
    image_names=splits["test"]
)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

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

# -------------------------
# Test best checkpoint
# -------------------------

model.load_state_dict(torch.load("best_model.pth", map_location=device))
model.eval()

test_accuracy = 0
test_dice = 0
test_iou = 0

with torch.no_grad():

    for images, masks in test_loader:

        images = images.to(device)
        masks = masks.to(device)

        preds = model(images)
        preds_bin = (torch.sigmoid(preds) > 0.5).float()

        correct = (preds_bin == masks).float().sum()
        test_accuracy += (correct / torch.numel(preds_bin)).item()

        metrics = evaluate_all(preds_bin.cpu().numpy(), masks.cpu().numpy())
        test_dice += metrics["Dice"]
        test_iou += metrics["IoU"]

test_accuracy /= len(test_loader)
test_dice /= len(test_loader)
test_iou /= len(test_loader)

print(f"Test Accuracy: {test_accuracy:.4f}")
print(f"Test Dice: {test_dice:.4f}")
print(f"Test IoU: {test_iou:.4f}")