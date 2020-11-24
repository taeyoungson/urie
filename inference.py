import json
import torch
import argparse

import numpy as np
import torch.nn as nn
import torch.utils.data as data
import torchvision.utils as vutils
import torchvision.models as models

from tqdm import tqdm

from models.skunet_model import SKUNet
from datasets.corruption_dataset import load_cub
from imagenet_dataset.imagenet_loader import load_imagenet


def validate(srcnn, classifier, dataloader, mode="cub", vis=False):
    srcnn.eval()
    classifier.eval()
    val_loss = 0
    clf_acc = 0
    acc = 0
    count = 0

    if mode == "cub":
        with torch.no_grad():
            for batch in tqdm(dataloader):
                if len(batch) == 3:
                    image, label, _ = batch
                else:
                    image, label = batch
                image = image.cuda()
                label = label.cuda()

                output = srcnn(image)

                if len(output) == 2:
                    # SKUNet case
                    model_out, _ = output
                else:
                    model_out = output

                pred = classifier(model_out)
                clf_only_pred = classifier(image)

                if vis:
                    vutils.save_image([image[0], model_out[0]], f"results/{count}.jpg", normalize=True, scale_each=True)

                count += 1

                loss = nn.CrossEntropyLoss()(pred, label)
                pred = pred.detach().cpu().numpy()
                clf_pred = clf_only_pred.detach().cpu().numpy()

                hit = np.count_nonzero(np.argmax(pred, axis=1) == label.detach().cpu().numpy())
                clf_hit = np.count_nonzero(np.argmax(clf_pred, axis=1) == label.detach().cpu().numpy())

                val_loss += loss.item()
                acc += hit
                clf_acc += clf_hit

        print(f"clf_only acc: {clf_acc / len(dataloader.dataset):.3f}")
        print(f"+enh  acc: {acc / len(dataloader.dataset):.3f}")
        return acc / len(dataloader.dataset)
    else:
        with torch.no_grad():
            for batch in tqdm(dataloader):
                if len(batch) == 3:
                    image, label, _ = batch
                else:
                    image, label = batch
                image = image.cuda()
                label = label.cuda()

                output = srcnn(image)
                if len(output) == 2:
                    # SKUNet case
                    model_out, _ = output
                else:
                    model_out = output

                pred = classifier(model_out)
                clf_only_pred = classifier(image)

                loss = nn.CrossEntropyLoss()(pred, label)
                pred = pred.detach().cpu().numpy()
                clf_pred = clf_only_pred.detach().cpu().numpy()

                hit = np.count_nonzero(np.argmax(pred, axis=1) == label.detach().cpu().numpy())
                clf_hit = np.count_nonzero(np.argmax(clf_pred, axis=1) == label.detach().cpu().numpy())

                val_loss += loss.item()
                acc += hit
                clf_acc += clf_hit

    print(f"clf_only acc: {clf_acc / len(dataloader.dataset):.3f}")
    print(f"+enh  acc: {acc / len(dataloader.dataset):.3f}")
    return acc / len(dataloader.dataset)


def main(args=None):
    if args.dataset == "cub":
        cln_dataloader = data.DataLoader(
            load_cub(crpMode="clean", tsfrmMode="eval"),
            batch_size=args.test_batch_size,
            shuffle=False,
            num_workers=32,
            pin_memory=True,
            drop_last=False,
        )

        tng_dataloader = data.DataLoader(
            load_cub(crpMode="train", tsfrmMode="eval"),
            batch_size=args.test_batch_size,
            shuffle=False,
            num_workers=32,
            pin_memory=True,
            drop_last=False,
        )

        val_dataloader = data.DataLoader(
            load_cub(crpMode="test", tsfrmMode="eval"),
            batch_size=args.test_batch_size,
            shuffle=False,
            num_workers=32,
            pin_memory=True,
            drop_last=False,
        )
    else:
        cln_dataloader = data.DataLoader(
            load_imagenet(crpMode="clean", tsfrmMode="eval"),
            batch_size=args.test_batch_size,
            shuffle=False,
            num_workers=32,
            pin_memory=True,
            drop_last=False,
        )

        tng_dataloader = data.DataLoader(
            load_imagenet(crpMode="train", tsfrmMode="eval"),
            batch_size=args.test_batch_size,
            shuffle=False,
            num_workers=32,
            pin_memory=True,
            drop_last=False,
        )

        val_dataloader = data.DataLoader(
            load_imagenet(crpMode="test", tsfrmMode="eval"),
            batch_size=args.test_batch_size,
            shuffle=False,
            num_workers=32,
            pin_memory=True,
            drop_last=False,
        )
    print(f"cln_dataloader load complete with ({len(cln_dataloader)}/{len(cln_dataloader.dataset)})")
    print(f"tng_dataloader load complete with ({len(tng_dataloader)}/{len(tng_dataloader.dataset)})")
    print(f"val_dataloader load complete with ({len(val_dataloader)}/{len(val_dataloader.dataset)})")

    new_weights = {}
    if args.enhancer == "ours" or args.enhancer == "mse":
        srcnn = SKUNet()
        weights = torch.load(args.srcnn_pretrained_path)

    elif args.enhancer == "owan":
        from owan_model import Network
        srcnn = Network(16, 10, nn.L1Loss())
        weights = torch.load(args.srcnn_pretrained_path)

    for k, v in weights.items():
        if "module." in k:
            new_weights[k.replace("module.", "")] = v
        else:
            new_weights[k] = v
    srcnn.load_state_dict(new_weights, strict=True)

    if args.mgpu:
        srcnn = nn.DataParallel(srcnn)
    srcnn.eval()
    srcnn.cuda()

    if args.recog == "r50":
        classifier = models.resnet50(pretrained=True)
    elif args.recog == "r101":
        classifier = models.resnet101(pretrained=True)
    elif args.recog == "v16":
        classifier = models.vgg16(pretrained=True)
    else:
        raise NotImplementedError

    if args.dataset == "cub":
        if args.recog == "v16":
            classifier.classifier[6] = nn.Linear(4096, 200)
            weights = torch.load("saved_models/base_models/vgg16_on_clean.ckpt.pt")

        elif args.recog == "r50":
            classifier.fc = nn.Linear(2048, 200)
            weights = torch.load("saved_models/base_models/resnet50_on_clean.ckpt.pt")

        elif args.recog == "r101":
            classifier.fc = nn.Linear(2048, 200)
            weights = torch.load("saved_models/base_models/resnet101_on_clean.ckpt.pt")

        new_weights = {}
        for k, v in weights.items():
            if "model." in k:
                new_weights[k.replace("model.", "")] = v
            else:
                new_weights[k] = v
        classifier.load_state_dict(new_weights, strict=True)

    if args.mgpu:
        classifier = nn.DataParallel(classifier)
    classifier.cuda()
    classifier.eval()

    print("... validating cln_dataloader")
    cln_acc = validate(srcnn, classifier, cln_dataloader, mode=args.dataset)
    print("... validating tng_dataloader")
    tng_acc = validate(srcnn, classifier, tng_dataloader, mode=args.dataset, vis=args.vis)
    print("... validating val_dataloader")
    val_acc = validate(srcnn, classifier, val_dataloader, mode=args.dataset)
    result_dict = {
        f"cln_acc": cln_acc,
        f"tng_acc": tng_acc,
        f"val_acc": val_acc,
    }
    print(result_dict)

    name = args.srcnn_pretrained_path.split("/")[-2]
    with open(f"metrics/{name}_{args.dataset}.json", "w") as fp:
        json.dump(result_dict, fp)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_batch_size", type=int, default=384)
    parser.add_argument("--srcnn_pretrained_path", type=str)
    parser.add_argument("--clf_pretrained_path", type=str)
    parser.add_argument("--residual", action="store_true")
    parser.add_argument("--tanh", action="store_true")
    parser.add_argument("--clf_only", action="store_true")
    parser.add_argument("--dataset", choices=["ilsvrc", "cub"])
    parser.add_argument("--enhancer", default="ours", choices=["ours", "owan", "mse"])
    parser.add_argument("--recog", required=True, choices=["r50", "r101", "v16"], type=str)
    parser.add_argument("--vis", action="store_true")
    parser.add_argument("--mgpu", action="store_true")


    parser.set_defaults(vis=False)
    parser.set_defaults(mgpu=False)
    args = parser.parse_args()
    print(f"loading model from {args.srcnn_pretrained_path}")

    main(args=args)
