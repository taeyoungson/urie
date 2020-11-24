import os
import csv
import torch
import random

import torch.utils.data as data
import torchvision.utils as vutils
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

from pathlib import Path

from datasets.utils import open_image
from datasets.utils import crop_to_bounding_box
from datasets.utils import restore_image_from_numpy
from datasets.corruption import corruptions


class DynamicCUB(data.Dataset):
    def __init__(self, **kwargs):
        super(DynamicCUB, self).__init__()
        self.crpModeChoices = ["train", "test", "clean", "mixed"]
        self.tsfrmModeChoices = ["train", "eval"]
        self.severityChoices = [1, 2, 3, 4, 5]

        self.crpMode = kwargs["crpMode"]
        self.tsfrmMode = kwargs["tsfrmMode"]
        self.verbose = kwargs["verbose"] if "verbose" in kwargs.keys() else False
        self.mixed_threshold = kwargs["threshold"] if "threshold" in kwargs.keys() else 0.5
        self.ten_crop_eval = kwargs["ten_crop_eval"] if "ten_crop_eval" in kwargs.keys() else False
        self.current_crp = None
        self.current_tsfrm = None

        self._init_corruption()
        self._init_transform()

        self.image_path = os.environ.get("CUB_IMAGE")
        self.distorted_image_path = os.environ.get("DISTORTED_CUB_IMAGE")
        self.tng_label_path = os.environ.get("CUB_TNG_LABEL")
        self.val_label_path = os.environ.get("CUB_VAL_LABEL")
        self.current_crp = self.crpMode2crps[self.crpMode]
        self.current_tsfrm = self.tsfrmMode2tsfrm[self.tsfrmMode]

        self.images = []
        self.bboxes = []

        self._load_images()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        imagepath, label = self.images[idx]
        # distorted_image_path = imagepath.replace(self.image_path, self.distorted_image_path)
        bbox = self.bboxes[idx]

        # print(f"loading from {imagepath}, with bbox {bbox}")
        image = open_image(imagepath)
        image = crop_to_bounding_box(image, bbox)

        if not self.crpMode == "clean":
            if self.tsfrmMode == "train":
                # when "train" is given for tsfrmMode
                crp_func = random.choice(self.current_crp)
                crp_idx = self.current_crp.index(crp_func)
                sev = random.choice(self.severityChoices)

                image = self.current_tsfrm(image)
                i, j, h, w = transforms.RandomCrop.get_params(
                    image, output_size=(227, 227))

                if crp_idx != 15:
                    # distorted_image_path = distorted_image_path.split("/")
                    distorted_image_path = Path(self.distorted_image_path)/ Path(crp_func) / Path(str(sev)) / "/".join(imagepath.split("/")[-2:])
                    # print(f"loading from {distorted_image_path}, with bbox {bbox}")
                    distorted_image = open_image(distorted_image_path)
                    distorted_image = crop_to_bounding_box(distorted_image, bbox)
                    distorted_image = self.current_tsfrm(distorted_image)

                else:
                    crp_idx = -1
                    distorted_image = image

                clean = TF.crop(image, i, j, h, w)
                distorted= TF.crop(distorted_image, i, j, h, w)

                if random.random() > 0.5:
                    clean = TF.hflip(clean)
                    distorted = TF.hflip(distorted)

                clean = self.finalizer_tsfrm(clean)
                distorted = self.finalizer_tsfrm(distorted)

                return distorted, clean, label, crp_idx

            else:
                # validation
                image = self.current_tsfrm(image)
                crp = imagepath.split("/")[4]

                return image, label, crp

        clean = self.current_tsfrm(image)
        if self.tsfrmMode == "train":
            i, j, h, w = transforms.RandomCrop.get_params(
                clean, output_size=(227, 227))
            clean = TF.crop(clean, i, j, h, w)
            if random.random() > 0.5:
                clean = TF.hflip(clean)
            clean = self.finalizer_tsfrm(clean)
        return clean, label

    def reconfig_dataset(self, crpMode, tsfrmMode):
        assert crpMode in self.crpModeChoices
        assert tsfrmMode in self.tsfrmModeChoices
        self.crpMode = crpMode
        self.tsfrmMode = tsfrmMode
        self.current_crp = self.crpMode2crps[self.crpMode]
        self.current_tsfrm = self.tsfrmMode2tsfrm[self.tsfrmMode]

        if self.verbose:
            print("current_crp : {}".format(self.current_crp))
            print("current_tsfrm : {}".format(self.current_tsfrm))

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
            "mixed": self.train_corruptions,
        }

    def _init_transform(self):
        self.train_tsfrm = transforms.Compose([
            transforms.Resize((256, 256)),
        ])
        self.finalizer_tsfrm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        if self.ten_crop_eval:
            self.eval_tsfrm = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.TenCrop((227, 227)),
                transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
                transforms.Lambda(lambda crops: torch.stack([transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                                  std=[0.229, 0.224, 0.225])(crop) for crop in crops])),
            ])
        else:
            self.eval_tsfrm = transforms.Compose([
                transforms.Resize((227, 227)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

        self.tsfrmMode2tsfrm = {
            "train": self.train_tsfrm,
            "eval": self.eval_tsfrm,
        }

    def _load_images(self):
        if self.tsfrmMode == "train" or self.crpMode == "clean":
            for row in self._read_csv():
                imagepath = os.path.join(self.image_path, row[0])
                label = int(row[5]) - 1

                self.images.append((imagepath, label))
                self.bboxes.append((int(row[1]), int(row[2]), int(row[3]), int(row[4])))
        else:
            for row in self._read_csv():
                imagepath = row[0]
                label = int(row[2]) - 1

                self.images.append((imagepath, label))
                self.bboxes.append(eval(row[1]))

    def _read_csv(self):
        if self.tsfrmMode == "train":
            label_path = self.tng_label_path
        elif self.crpMode == "clean":
            label_path = self.val_label_path
        else:
            label_path = os.environ.get(f"CUB_TNG_{self.crpMode.upper()}_VAL")

        # print(f"loading form {label_path}")
        with open(label_path) as csv_file:
            csv_file_rows = csv.reader(csv_file, delimiter=",")
            for row in csv_file_rows:
                yield row


def load_cub(**kwargs):
    return DynamicCUB(**kwargs)
