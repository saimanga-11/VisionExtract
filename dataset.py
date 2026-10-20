import os
import cv2
import torch
from torch.utils.data import Dataset


class SegmentationDataset(Dataset):

    def __init__(self, image_dir, mask_dir, size=(256, 256), image_names=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.size = size
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
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Read mask
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"Unable to read mask: {mask_path}")

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