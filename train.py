import os
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
    image_names=splits["train"],
    augment=True,
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

train_loader = DataLoader(
    train_dataset, batch_size=16, shuffle=True, num_workers=2, pin_memory=True
)
val_loader = DataLoader(
    val_dataset, batch_size=16, shuffle=False, num_workers=2, pin_memory=True
)
test_loader = DataLoader(
    test_dataset, batch_size=16, shuffle=False, num_workers=2, pin_memory=True
)

# -------------------------
# Model
# -------------------------

model = smp.Unet(
    encoder_name="resnet34",
    encoder_weights=None,
    in_channels=3,
    classes=1
).to(device)

bce_loss_fn = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="max",
    factor=0.5,
    patience=2,
)

epochs = 15

def dice_loss(logits, masks, smooth=1.0):
    probabilities = torch.sigmoid(logits)
    probabilities = probabilities.flatten(start_dim=1)
    masks = masks.flatten(start_dim=1)
    intersection = (probabilities * masks).sum(dim=1)
    dice = (2 * intersection + smooth) / (
        probabilities.sum(dim=1) + masks.sum(dim=1) + smooth
    )
    return 1 - dice.mean()


def evaluate(loader):
    model.eval()
    totals = {
        "IoU": 0.0,
        "Dice": 0.0,
        "Precision": 0.0,
        "Recall": 0.0,
        "Accuracy": 0.0,
    }

    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device)
            masks = masks.to(device)
            predictions = (torch.sigmoid(model(images)) > 0.5).float()
            metrics = evaluate_all(predictions.cpu().numpy(), masks.cpu().numpy())
            for name, value in metrics.items():
                totals[name] += value

    return {name: value / len(loader) for name, value in totals.items()}


best_val_iou = -1.0
best_epoch = 0

if os.path.exists("best_model.pth"):
    model.load_state_dict(torch.load("best_model.pth", map_location=device))
    initial_metrics = evaluate(val_loader)
    best_val_iou = initial_metrics["IoU"]
    print(f"Resuming from best_model.pth (validation IoU: {best_val_iou:.4f})")

# -------------------------
# Training Loop
# -------------------------

for epoch in range(epochs):

    model.train()

    train_loss = 0
    for images, masks in train_loader:

        images = images.to(device)
        masks = masks.to(device)

        preds = model(images)

        loss = bce_loss_fn(preds, masks) + dice_loss(preds, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(train_loader)

    # -------------------------
    # Validation
    # -------------------------

    val_metrics = evaluate(val_loader)
    scheduler.step(val_metrics["IoU"])

    # -------------------------
    # Print results
    # -------------------------

    print(f"Epoch {epoch+1}/{epochs}")
    print(f"Train Loss: {train_loss:.4f}")
    print(f"Validation IoU: {val_metrics['IoU']:.4f}")
    print(f"Validation Dice: {val_metrics['Dice']:.4f}")
    print(f"Validation Precision: {val_metrics['Precision']:.4f}")
    print(f"Validation Recall: {val_metrics['Recall']:.4f}")
    print(f"Validation Accuracy: {val_metrics['Accuracy']:.4f}")
    print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.6g}")
    print("--------------------------------------------------")

    # -------------------------
    # Save best model
    # -------------------------

    if val_metrics["IoU"] > best_val_iou:
        best_val_iou = val_metrics["IoU"]
        best_epoch = epoch + 1
        torch.save(model.state_dict(), "best_model.pth")

# -------------------------
# Training Complete
# -------------------------

print("Training Finished")
print(f"Best Epoch: {best_epoch}")
print(f"Best Validation IoU: {best_val_iou:.4f}")

# -------------------------
# Test best checkpoint
# -------------------------

model.load_state_dict(torch.load("best_model.pth", map_location=device))
test_metrics = evaluate(test_loader)
for name, value in test_metrics.items():
    print(f"Test {name}: {value:.4f}")