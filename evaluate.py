import cv2
import numpy as np
import segmentation_models_pytorch as smp
import torch
from pathlib import Path
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader

from dataset import SegmentationDataset
from metrics import evaluate_all
from prepare_data import DATASET_DIR, ROOT_DIR, prepare_data


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
METRIC_NAMES = ("IoU", "Dice", "Precision", "Recall", "Accuracy")


def load_model():
    checkpoint_path = ROOT_DIR / "best_model.pth"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")

    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=3,
        classes=1,
    ).to(DEVICE)
    model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
    model.eval()
    return model


def retain_examples(examples, example, best):
    examples.append(example)
    examples.sort(key=lambda item: (item["iou"], item["name"]), reverse=best)
    del examples[3:]


def render_example(example, output_path, label):
    image = cv2.imread(str(example["image_path"]), cv2.IMREAD_COLOR)
    mask = cv2.imread(str(example["mask_path"]), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Unable to read image: {example['image_path']}")
    if mask is None:
        raise ValueError(f"Unable to read mask: {example['mask_path']}")

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height, width = image.shape[:2]
    if mask.shape != (height, width):
        mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
    ground_truth = np.where(mask > 127, 255, 0).astype(np.uint8)
    predicted_probability = cv2.resize(
        example["probability"],
        (width, height),
        interpolation=cv2.INTER_LINEAR,
    )
    predicted_mask = np.where(predicted_probability > 0.5, 255, 0).astype(np.uint8)
    isolated = image * (predicted_mask[:, :, None] > 0)

    panel_width = 360
    panel_height = 320
    header_height = 36
    panels = [
        ("Original", image),
        ("Ground truth", ground_truth),
        ("Prediction", predicted_mask),
        ("Isolated output", isolated),
    ]
    canvas = Image.new(
        "RGB",
        (panel_width * len(panels), panel_height + header_height + 28),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    for index, (title, pixels) in enumerate(panels):
        panel = Image.fromarray(pixels)
        panel.thumbnail((panel_width, panel_height))
        x = index * panel_width
        y = header_height + (panel_height - panel.height) // 2
        canvas.paste(panel, (x, y))
        draw.text((x + 8, 10), title, fill="black")

    draw.text(
        (8, panel_height + header_height + 6),
        f"{label} | {example['name']} | IoU: {example['iou']:.4f}",
        fill="black",
    )
    canvas.save(output_path)


def main():
    splits = prepare_data()
    test_image_dir = DATASET_DIR / "DUTS-TE" / "DUTS-TE-Image"
    test_mask_dir = DATASET_DIR / "DUTS-TE" / "DUTS-TE-Mask"
    test_dataset = SegmentationDataset(
        str(test_image_dir),
        str(test_mask_dir),
        image_names=splits["test"],
    )
    if not test_dataset:
        raise ValueError("The test split contains no matched image and mask pairs.")

    model = load_model()
    loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    metric_totals = {name: 0.0 for name in METRIC_NAMES}
    best_examples = []
    worst_examples = []

    with torch.inference_mode():
        for batch_index, (images, masks) in enumerate(loader):
            probabilities = torch.sigmoid(model(images.to(DEVICE))).cpu().numpy()
            masks = masks.numpy()
            start = batch_index * loader.batch_size

            for offset, (probability, target) in enumerate(zip(probabilities, masks)):
                sample_index = start + offset
                image_name, mask_name = test_dataset.samples[sample_index]
                predicted = probability > 0.5
                scores = evaluate_all(predicted, target)
                for name in METRIC_NAMES:
                    metric_totals[name] += scores[name]

                example = {
                    "name": image_name,
                    "iou": scores["IoU"],
                    "probability": probability[0],
                    "image_path": test_image_dir / image_name,
                    "mask_path": test_mask_dir / mask_name,
                }
                retain_examples(best_examples, example, best=True)
                retain_examples(worst_examples, example, best=False)

    print("Test metrics (per-image scores averaged over the test split):")
    for name in METRIC_NAMES:
        print(f"Average {name}: {metric_totals[name] / len(test_dataset):.4f}")

    docs_dir = ROOT_DIR / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)
    for rank, example in enumerate(best_examples, start=1):
        filename = f"best_{rank}_{Path(example['name']).stem}.png"
        render_example(example, docs_dir / filename, f"Best {rank}")
    for rank, example in enumerate(worst_examples, start=1):
        filename = f"worst_{rank}_{Path(example['name']).stem}.png"
        render_example(example, docs_dir / filename, f"Worst {rank}")
    print(f"Saved {len(best_examples) + len(worst_examples)} examples to {docs_dir}")


if __name__ == "__main__":
    main()
