import os
import csv
import random

import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

from pathlib import Path

from datasets.utils import open_image
from datasets.utils import restore_image_from_numpy
from datasets.corruption import corruptions
from imagenet_dataset.utils import parse_label_id


class IMAGENET_DATASET(data.Dataset):
    def __init__(self, **kwargs):
        super(IMAGENET_DATASET).__init__()
        self.crpModeChoices = ["train", "test", "clean"]
        self.tsfrmModeChoices = ["train", "eval"]
        self.severityChoices = [1, 2, 3, 4, 5]

        self.crpMode = kwargs["crpMode"]
        self.tsfrmMode = kwargs["tsfrmMode"]

        self.current_crp = None
        self.current_tsfrm = None

        self._init_corruption()
        self._init_transform()

        self.imagenet_root = os.environ.get("IMAGENET_ROOT")
        self.imagenet_c_root = os.environ.get("IMAGENET_C_ROOT")

        self.tng_cln_csv = os.environ.get("IMAGENET_CLN_TNG_CSV")

        self.val_cln_csv = os.environ.get("IMAGENET_CLN_VAL_CSV")
        self.val_tng_csv = os.environ.get("IMAGENET_TNG_VAL_CSV")
        self.val_val_csv = os.environ.get("IMAGENET_VAL_VAL_CSV")

        self.current_crp = self.crpMode2crps[self.crpMode]
        self.current_tsfrm = self.tsfrmMode2tsfrm[self.tsfrmMode]
        self.images = []

        self._load_images()

    def __len__(self):
        # for debug
        # return 112

        return len(self.images)

    def __getitem__(self, index):
        imagepath, label = self.images[index]

        # if self.imagenet_c_root in imagepath:
            # crp = imagepath.split("/")[4]
            # crp_func = corruptions[crp]
            # imagepath = imagepath.replace(self.imagenet_c_root, self.imagenet_root)
            # imagepath = imagepath.split("/")
            # imagepath = "/".join(imagepath[:4] + imagepath[6:])

        image = open_image(imagepath)

        if self.tsfrmMode == "train":
            # training part
            crp_func = random.choice(self.current_crp)
            crp_idx = self.current_crp.index(crp_func)
            sev = random.choice([1, 2, 3, 4, 5])

            image = self.current_tsfrm(image)
            i, j, h, w = transforms.RandomCrop.get_params(
                image, output_size=(224, 224))

            if crp_idx != 15:
                # if not clean image
                crp_func = corruptions[crp_func]
                distorted_image = open_image(imagepath)
                distorted_image = crp_func(distorted_image, sev)
                distorted_image = restore_image_from_numpy(distorted_image)
                distorted_image = self.current_tsfrm(distorted_image)

            else:
                # return clean image
                distorted_image = image

            clean = TF.crop(image, i, j, h, w)
            distorted = TF.crop(distorted_image, i, j, h, w)

            if random.random() > 0.5:
                clean = TF.hflip(clean)
                distorted = TF.hflip(distorted)

            clean = self.finalizer_tsfrm(clean)
            distorted = self.finalizer_tsfrm(distorted)

            return distorted, clean, label, crp_idx

        else:
            image = self.current_tsfrm(image)
            image = self.finalizer_tsfrm(image)
            return image, label

    def _init_corruption(self):
        corruptions = ["gaussian_noise", "shot_noise", "impulse_noise",
                       "defocus_blur", "glass_blur", "motion_blur", "zoom_blur",
                       "snow", "frost", "fog",
                       "brightness", "contrast", "elastic_transform", "pixelate", "jpeg_compression", "clean",
                       "speckle_noise", "gaussian_blur", "spatter", "saturate"]

        self.train_corruptions = corruptions[:16]
        self.test_corruptions = corruptions[16:]
        self.all_corruptions = corruptions

        self.crpMode2crps = {
            "train": self.train_corruptions,
            "test": self.test_corruptions,
            "clean": None,
        }

    def _init_transform(self):
        self.train_tsfrm = transforms.Compose([
            transforms.Resize((256, 256)),
        ])
        self.finalizer_tsfrm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.eval_tsfrm = transforms.Compose([
            transforms.Resize((224, 224)),
        ])

        self.tsfrmMode2tsfrm = {
            "train": self.train_tsfrm,
            "eval": self.eval_tsfrm,
        }

    def _read_csv(self):
        if self.tsfrmMode == "train":
            label_path = self.tng_cln_csv

        elif self.crpMode == "clean":
            label_path = self.val_cln_csv

        elif self.crpMode == "train":
            label_path = self.val_tng_csv

        elif self.crpMode == "test":
            label_path = self.val_val_csv

        print(f"loading form {label_path}")

        with open(label_path) as csv_file:
            csv_file_rows = csv.reader(csv_file, delimiter=",")
            for row in csv_file_rows:
                yield row

    def _load_images(self):
        for row in self._read_csv():
            imagepath = os.path.join(self.imagenet_root, row[0])
            label = int(row[1])

            self.images.append((imagepath, label))

def load_imagenet(**kwargs):
    return IMAGENET_DATASET(**kwargs)
