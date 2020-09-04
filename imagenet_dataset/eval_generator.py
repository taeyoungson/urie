import os
import random
import numpy as np

from tqdm import tqdm
from pathlib import Path

from utils import parse_label_id


def main():
    # image_root = Path(os.environ.get("IMAGENET_ROOT"))
    image_root = Path(os.environ.get("IMAGENET_VALIDATION_ROOT"))
    label_dict = parse_label_id()

    # corruption_image_root = Path(os.environ.get("IMAGENET_C_ROOT"))
    corruption_image_root = Path(os.environ.get("IMAGENET_C_VALIDATION_ROOT"))

    # corruptions = ["gaussian_noise", "shot_noise", "impulse_noise",
                    # "defocus_blur", "glass_blur", "motion_blur", "zoom_blur",
                    # "snow", "frost", "fog",
                    # "brightness", "contrast", "elastic_transform", "pixelate", "jpeg_compression",]
    corruptions = ["speckle_noise", "gaussian_blur", "spatter", "saturate"]
    severity = ["3"]

    with open("imagenet_val_tsfrm_validation.csv", "w") as f:
        # for validation script generation
        for folder in image_root.iterdir():
            for image_path in tqdm(folder.iterdir()):
                image_name = Path(image_path.as_posix().split("/")[-1])

                # for validation script generation
                label = label_dict[folder.as_posix().split("/")[-1]]

                # label = label_dict[image_name.as_posix().split("_")[0]]
                crp = Path(random.choice(corruptions))
                sev = Path("3")

                image_full_path = corruption_image_root / crp / sev / folder / image_name

                # for clean csv generation
                # f.write(f"{image_path.as_posix()},{label}\n")

                # for tsfrmed version csv generation
                f.write(f"{image_full_path.as_posix()},{label}\n")


if __name__ == "__main__":
    main()
