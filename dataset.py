import os
import random
import cv2
import numpy as np
import torch
from PIL import Image as PILImage
from torch.utils.data import Dataset
from torchvision.transforms import ColorJitter


class SegmentationDataset(Dataset):

    def __init__(
        self,
        image_dir,
        mask_dir,
        size=(256, 256),
        image_names=None,
        augment=False,
    ):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.size = size
        self.augment = augment
        self.color_jitter = ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=0.05,
        )
        candidate_names = image_names
        if candidate_names is None:
            candidate_names = sorted(
                name for name in os.listdir(image_dir)
                if name.lower().endswith(".jpg")
            )

        mask_names = {
            os.path.splitext(name)[0]: name
            for name in os.listdir(mask_dir)
            if name.lower().endswith(".png")
        }
        self.samples = []
        missing_masks = []
        for image_name in candidate_names:
            mask_name = mask_names.get(os.path.splitext(image_name)[0])
            if mask_name is None:
                missing_masks.append(image_name)
                continue
            self.samples.append((image_name, mask_name))

        if missing_masks:
            print(
                f"Skipping {len(missing_masks)} image(s) without matching masks "
                f"in {image_dir}: {', '.join(missing_masks)}"
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):

        img_name, mask_name = self.samples[index]

        img_path = os.path.join(self.image_dir, img_name)
        mask_path = os.path.join(self.mask_dir, mask_name)

        # Read image
        image = cv2.imread(img_path)
        if image is None:
            raise ValueError(f"Unable to read image: {img_path}")

        # Read mask
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"Unable to read mask: {mask_path}")

        if self.augment:
            if random.random() < 0.5:
                image = cv2.flip(image, 1)
                mask = cv2.flip(mask, 1)

            height, width = image.shape[:2]
            crop_height = random.randint(max(1, int(height * 0.8)), height)
            crop_width = random.randint(max(1, int(width * 0.8)), width)
            top = random.randint(0, height - crop_height)
            left = random.randint(0, width - crop_width)
            image = image[top:top + crop_height, left:left + crop_width]
            mask = mask[top:top + crop_height, left:left + crop_width]

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = np.asarray(self.color_jitter(PILImage.fromarray(image)))
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Resize (VERY IMPORTANT for model consistency)
        image = cv2.resize(image, self.size)
        mask = cv2.resize(mask, self.size, interpolation=cv2.INTER_NEAREST)

        # Normalize
        image = image.astype("float32") / 255.0
        mask = (mask > 127).astype("float32")

        # Convert to tensor
        image = torch.tensor(image).permute(2, 0, 1)   # (C, H, W)
        mask = torch.tensor(mask).unsqueeze(0)         # (1, H, W)

        return image, mask