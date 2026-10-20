import json
import random
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent
DATASET_DIR = ROOT_DIR / "dataset"
SEED = 42


def matched_image_names(image_dir, mask_dir):
    mask_stems = {path.stem for path in mask_dir.glob("*.png")}
    image_names = []
    missing_masks = []

    for image_path in sorted(image_dir.glob("*.jpg")):
        if image_path.stem in mask_stems:
            image_names.append(image_path.name)
        else:
            missing_masks.append(image_path.name)

    if missing_masks:
        print(
            f"Skipping {len(missing_masks)} image(s) without matching masks "
            f"in {image_dir}: {', '.join(missing_masks)}"
        )
    return image_names


def prepare_data():
    train_image_dir = DATASET_DIR / "DUTS-TR" / "DUTS-TR-Image"
    train_mask_dir = DATASET_DIR / "DUTS-TR" / "DUTS-TR-Mask"
    test_image_dir = DATASET_DIR / "DUTS-TE" / "DUTS-TE-Image"
    test_mask_dir = DATASET_DIR / "DUTS-TE" / "DUTS-TE-Mask"

    train_names = matched_image_names(train_image_dir, train_mask_dir)
    test_names = matched_image_names(test_image_dir, test_mask_dir)

    random.Random(SEED).shuffle(train_names)
    train_count = int(len(train_names) * 0.9)
    splits = {
        "train": train_names[:train_count],
        "val": train_names[train_count:],
        "test": test_names,
    }

    splits_path = DATASET_DIR / "splits.json"
    splits_path.write_text(json.dumps(splits, indent=2) + "\n", encoding="utf-8")
    print(
        f"Prepared splits: {len(splits['train'])} train, "
        f"{len(splits['val'])} validation, {len(splits['test'])} test"
    )
    return splits


if __name__ == "__main__":
    prepare_data()
