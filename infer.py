import argparse
from pathlib import Path

import cv2
import numpy as np
import segmentation_models_pytorch as smp
import torch

from prepare_data import ROOT_DIR


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_SIZE = (256, 256)


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


def main():
    parser = argparse.ArgumentParser(
        description="Isolate the subject in an image using the trained model."
    )
    parser.add_argument("input_path", type=Path, help="Path to the input image")
    parser.add_argument("output_path", type=Path, help="Path for the isolated image")
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Keep only the largest connected region of the predicted mask",
    )
    args = parser.parse_args()

    image = cv2.imread(str(args.input_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Unable to read input image: {args.input_path}")

    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb_image, IMAGE_SIZE).astype(np.float32) / 255.0
    input_tensor = (
        torch.from_numpy(resized)
        .permute(2, 0, 1)
        .unsqueeze(0)
        .to(DEVICE)
    )

    model = load_model()
    with torch.inference_mode():
        probability = torch.sigmoid(model(input_tensor))[0, 0].cpu().numpy()

    height, width = image.shape[:2]
    probability = cv2.resize(
        probability, (width, height), interpolation=cv2.INTER_LINEAR
    )
    mask = (probability > 0.5).astype(np.uint8)

    if args.clean and np.any(mask):
        component_count, labels, stats, _ = cv2.connectedComponentsWithStats(
            mask, connectivity=8
        )
        largest_label = 1 + np.argmax(stats[1:component_count, cv2.CC_STAT_AREA])
        mask = (labels == largest_label).astype(np.uint8)

    isolated = image * mask[:, :, None]
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(args.output_path), isolated):
        raise ValueError(f"Unable to save output image: {args.output_path}")
    print(f"Saved isolated image to {args.output_path}")


if __name__ == "__main__":
    main()
