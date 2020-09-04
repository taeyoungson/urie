import wandb
import torch
import argparse

import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.models as models

from pathlib import Path
from tqdm import tqdm

from voc_dataset.voc_loader import load_voc
from voc_dataset.metrics import mean_average_precision
from datasets.corruption_dataset import load_cub


def main(args):
    model = models.resnet18(pretrained=True)
    if args.dataset == "cub":
        model.fc = nn.Linear(512, 200)
    else:
        model.fc = nn.Linear(512, 20)

    # model = models.vgg16(pretrained=True)
    # model.classifier[6] = nn.Linear(4096, 200)

    model.cuda()
    wandb.init(config=args, project="cv-project", name=args.desc)
    wandb.watch(model)

    params_list = []
    params_list.append({"params": model.conv1.parameters()})
    params_list.append({"params": model.bn1.parameters()})
    params_list.append({"params": model.layer1.parameters()})
    params_list.append({"params": model.layer2.parameters()})
    params_list.append({"params": model.layer3.parameters()})
    params_list.append({"params": model.layer4.parameters()})
    params_list.append({"params": model.fc.parameters(), "lr": args.lr * 10})

    # Use the follwing lines to train vgg baseline
    # params_list.append({"params": model.features.parameters()})
    # params_list.append({"params": model.classifier[0].parameters()})
    # params_list.append({"params": model.classifier[3].parameters()})
    # params_list.append({"params": model.classifier[6].parameters(), "lr":args.lr * 10})

    optimizer = optim.Adam(params_list, lr=args.lr, weight_decay=1e-3)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, 20)

    if args.dataset == "cub":
        print("Training with CUB-200-2011")
        tng_dataloader = data.DataLoader(
            load_cub(crpMode="clean", tsfrmMode="train"),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=16,
            pin_memory=True,
            drop_last=False,
        )

        val_dataloader = data.DataLoader(
            load_cub(crpMode="train", tsfrmMode="eval", ten_crop_eval=False),
            batch_size=args.test_batch_size,
            shuffle=False,
            num_workers=16,
            pin_memory=True,
            drop_last=False,
        )
        criterion = nn.CrossEntropyLoss().cuda()
    elif args.dataset == "voc":
        print("Training with PASCAL VOC DATASET")
        criterion = nn.BCELoss().cuda()
        tng_dataloader = load_voc(type="train", corruption="clean")
        val_dataloader = load_voc(type="val", corruption="clean")
    print(f"tng_dataloader load complete with ({len(tng_dataloader)}/{len(tng_dataloader.dataset)})")
    print(f"val_dataloader load complete with ({len(val_dataloader)}/{len(val_dataloader.dataset)})")


    model.train()
    for e in range(args.epochs):
        # loss terms here
        loss = 0.

        for i, batch in tqdm(enumerate(tng_dataloader)):
            # training here
            image, label = batch
            image = image.cuda()
            # for cub
            # label = label.cuda() - 1

            label = label.cuda().float()

            optimizer.zero_grad()

            pred = model(image)
            pred = nn.Sigmoid()(pred)

            clf_loss = criterion(pred, label)
            loss += clf_loss.item()

            clf_loss.backward()
            optimizer.step()

            print("===> Epoch[{}]({}/{}): clf_loss : {:.4f}"
                .format(e, i, len(tng_dataloader), clf_loss.item()))

        wandb.log({"clf_loss": loss / len(tng_dataloader)})
        lr_scheduler.step()

        if e % args.val_period != 0:
            continue

        val_loss = 0.
        best_map = 0.
        acc = 0
        model.eval()
        pred_whole = None
        with torch.no_grad():
            for i, batch in tqdm(enumerate(val_dataloader)):
                # validation here
                if len(batch) == 2:
                    image, label = batch
                else:
                    image, label, _ = batch
                image = image.cuda()
                # for cub
                # label = label.cuda() - 1

                # for voc
                label = label.cuda().float()

                # B, _, C, H, W = image.shape
                # image = image.view(-1, C, H, W)
                pred = model(image)
                pred = nn.Sigmoid()(pred)

                # for mAP calculation for voc

                loss = criterion(pred, label)
                pred = pred.detach().cpu().numpy()
                label = label.detach().cpu().numpy()
                if pred_whole is not None:
                    pred_whole = np.concatenate((pred_whole, pred))
                    label_whole = np.concatenate((label_whole, label))

                else:
                    pred_whole = pred
                    label_whole = label
                # hit = np.count_nonzero(np.argmax(pred, axis=1) == label.detach().cpu().numpy())

                val_loss += loss.item()
                # acc += hit

        if args.dataset == "cub":
            wandb.log({"val_loss": val_loss / len(val_dataloader), "acc": acc / len(val_dataloader.dataset)})

        else:
            mAP = mean_average_precision(pred_whole, label_whole)
            if best_map < mAP:
                best_map = mAP
                if not Path.exists(Path(f"saved_models/base_models/").resolve()):
                    Path.mkdir(Path(f"saved_models/base_models").resolve())

                print(f"model saved to (saved_models/base_models/resnet18_on_VOC_clean_{best_map:2f}.ckpt.pt)")
                torch.save(model.state_dict(), f"saved_models/base_models/resnet18_on_VOC_clean_{best_map:2f}.ckpt.pt")

            wandb.log({"val_loss": val_loss / len(val_dataloader), "mAP": mAP})
        model.train()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=2, help='number of epochs to train for')
    parser.add_argument('--batch_size', type=int, default=32, help='training batch size')
    parser.add_argument('--test_batch_size', type=int, default=10, help='testing batch size')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning Rate. Default=0.01')
    parser.add_argument("--desc", type=str, default="no description")
    parser.add_argument("--val_period", type=int, default=1)
    parser.add_argument("--dataset", type=str, choices=["cub", "voc"])
    args = parser.parse_args()
    main(args)
