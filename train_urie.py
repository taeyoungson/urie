from __future__ import print_function
import json
import argparse
import wandb

import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm
from torch.utils.data import DataLoader

from datasets.corruption_dataset import load_cub
from imagenet_dataset.imagenet_loader import load_imagenet
import torch.utils.data as data
from utils import plot_images_to_wandb
from metrics import count_match

import torch.nn.functional as F
from pathlib import Path
import torchvision.models as models
from models.skunet_model import SKUNet


parser = argparse.ArgumentParser(description='PyTorch Super Res Example')
parser.add_argument('--batch_size', type=int, default=64, help='training batch size')
parser.add_argument('--test_batch_size', type=int, default=10, help='testing batch size')
parser.add_argument('--epochs', type=int, default=2, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.01, help='Learning Rate. Default=0.01')
parser.add_argument('--cuda', action='store_true', help='use cuda?')
parser.add_argument('--threads', type=int, default=4, help='number of threads for data loader to use')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
parser.add_argument("--desc", type=str, default="no description")
parser.add_argument("--save", type=str, default="remains")
parser.add_argument("--tanh", action="store_true")
parser.add_argument("--e2e", action="store_true")
parser.add_argument("--residual", action="store_true")
parser.add_argument("--classifier_tuning", action="store_true")
parser.add_argument("--load_srcnn", type=str, default=None)
parser.add_argument("--load_classifier", action="store_true")
parser.add_argument("--multi", action="store_true")
parser.add_argument("--weight_decay", type=float, default=1e-3)
parser.add_argument("--dataset", type=str, required=True, choices=["ilsvrc", "cub"])
parser.add_argument("--backbone", type=str, choices=["r18", "r50"])

parser.set_defaults(tanh=False)
parser.set_defaults(multi=False)
parser.set_defaults(e2e=False)
parser.set_defaults(residual=False)
parser.set_defaults(classifier_tuning=False)
parser.set_defaults(load_classifier=False)
opt = parser.parse_args()

torch.manual_seed(opt.seed)
torch.cuda.manual_seed(opt.seed)

if opt.dataset.lower() == "ilsvrc":
    tng_dataloader = data.DataLoader(
        load_imagenet(crpMode="train", tsfrmMode="train"),
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=36,
        pin_memory=True,
        drop_last=False,
    )

    val_dataloader = data.DataLoader(
        load_imagenet(crpMode="train", tsfrmMode="eval"),
        batch_size=opt.test_batch_size,
        shuffle=False,
        num_workers=36,
        pin_memory=True,
        drop_last=False,
    )
elif opt.dataset.lower() == "cub":
    tng_dataloader = data.DataLoader(
        load_cub(crpMode="train", tsfrmMode="train"),
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=36,
        pin_memory=True,
        drop_last=False,
    )

    val_dataloader = data.DataLoader(
        load_cub(crpMode="train", tsfrmMode="eval"),
        batch_size=opt.test_batch_size,
        shuffle=False,
        num_workers=36,
        pin_memory=True,
        drop_last=False,
    )
else:
    raise NotImplementedError

criterion = nn.CrossEntropyLoss().cuda()
print(f"tng_dataloader load complete with ({len(tng_dataloader)}/{len(tng_dataloader.dataset)})")
print(f"val_dataloader load complete with ({len(val_dataloader)}/{len(val_dataloader.dataset)})")

criterion = criterion.cuda()

srcnn = SKUNet()
srcnn.cuda()

if opt.backbone.lower() == "r18":
    classifier = models.resnet18(pretrained=True)

elif opt.backbone.lower() == "r50":
    classifier = models.resnet50(pretrained=True)

else:
    raise NotImplementedError

# lets fix classifier!

if opt.load_classifier and opt.dataset.lower() == "cub":
    print(f"Using classifier trained on CUB")

    if opt.backbone.lower() == "r18":
        pretrained_weights = torch.load("./base_models/resnet18_on_clean.ckpt.pt")
        classifier.fc = nn.Linear(512, 200)
    elif opt.backbone.lower() == "r50":
        pretrained_weights = torch.load("./base_models/resnet50_on_clean.ckpt.pt")
        classifier.fc = nn.Linear(2048, 200)
    classifier.load_state_dict(pretrained_weights, strict=True)
else:
    # ilsvrc2012 training
    print(f"Using classifier trained on ILSVRC2012")




params_list = []
params_list.append({"params": srcnn.parameters()})

print("...training enhancement layer only")
optimizer = optim.Adam(srcnn.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)

lr_scheduler = optim.lr_scheduler.StepLR(optimizer, 10)
classifier.eval().cuda()

if opt.multi:
    classifier = nn.DataParallel(classifier)
    srcnn = nn.DataParallel(srcnn)

import time


def train(epoch):
    epoch_loss = 0
    mse_total = 0.
    clf_total = 0
    srcnn.train()
    hit = 0
    for iteration, batch in enumerate(tng_dataloader, 1):
        start = time.time()
        input, target, label, corruption_idx = batch
        input = input.cuda()
        target = target.cuda()
        label = label.cuda()

        optimizer.zero_grad()
        model_out, residual = srcnn(input)
        clf_pred = classifier(model_out)

        clf_loss = criterion(clf_pred, label)

        hit += count_match(clf_pred, label)
        mse_loss = nn.MSELoss()(model_out, target)

        loss = clf_loss

        clf_total += clf_loss.item()
        mse_total += mse_loss.item()
        epoch_loss += clf_loss.item() + mse_loss.item()
        loss.backward()
        optimizer.step()

        print("===> Epoch[{}]({}/{}): clf_loss : {:.4f}, iter: {:.3f}"
              .format(epoch, iteration, len(tng_dataloader), clf_loss.item(), time.time() - start))
        # wandb.log({"iteration_loss": clf_loss.item()})

    print("===> Epoch {} Complete: Avg. Loss: {:.4f}".format(epoch, epoch_loss / len(tng_dataloader)))

    plot_images_to_wandb([input[0], model_out[0], residual[0], target[0]], "Comparison", step=epoch)
    wandb.log({
        "train_acc": hit / len(tng_dataloader.dataset),
        "clf_loss": clf_total / len(tng_dataloader),
        "mse_loss": mse_total / len(tng_dataloader),
        "total_loss": epoch_loss / len(tng_dataloader),
    }, step=epoch)
    lr_scheduler.step()

def validate(epoch):
    srcnn.eval()
    classifier.eval()
    val_loss = 0
    acc = 0

    with torch.no_grad():
        for batch in tqdm(val_dataloader):
            image, label, corruption = batch

            image = image.cuda()
            label = label.cuda()

            # with corruption detection
            model_out, residual = srcnn(image)
            pred = classifier(model_out)

            loss = nn.CrossEntropyLoss()(pred, label)

            hit = count_match(pred, label)

            val_loss += loss.item()
            acc += hit

    wandb.log({"val_loss": val_loss / len(val_dataloader), "acc": acc / len(val_dataloader.dataset)}, step=epoch)
    return acc / len(val_dataloader.dataset)


def checkpoint(acc, name):
    if not Path.exists(Path(f"saved_models/{name}/").resolve()):
        Path.mkdir(Path(f"saved_models/{name}").resolve())

    for f in Path(f"saved_models/{name}").iterdir():
        if f.as_posix().endswith("pth"):
            Path.unlink(f)

    model_out_path = f"saved_models/{name}/srcnn_acc_{acc:.3f}.pth"
    classifier_out_path = f"saved_models/{name}/classifier_acc_{acc:.3f}.pth"

    torch.save(srcnn.state_dict(), model_out_path)
    torch.save(classifier.state_dict(), classifier_out_path)

    print("Checkpoint saved to {}".format(model_out_path))

def give_me_acc(srcnn, classifier, dataloader, plot_acc=False):
    srcnn.eval()
    classifier.eval()
    val_loss = 0
    acc = 0
    result_hit = {}
    result_total = {}
    result_acc = {}
    with torch.no_grad():
        for batch in tqdm(dataloader):
            if plot_acc:
                image, label, corruption = batch
            else:
                image, label = batch

            image = image.cuda()
            # changed for ilsvrc training, if you train cub, need to be changed
            label = label.cuda()

            model_out, _ = srcnn(image)
            pred = classifier(model_out)

            loss = nn.CrossEntropyLoss()(pred, label)
            pred = pred.detach().cpu().numpy()

            equals = (np.argmax(pred, axis=1) == label.detach().cpu().numpy())
            hit = np.count_nonzero(np.argmax(pred, axis=1) == label.detach().cpu().numpy())

    if plot_acc:
        for h, t in zip(result_hit.items(), result_total.items()):
            hk, hv = h
            tk, tv = t

            result_acc[hk] = hv / tv
        return acc / len(dataloader.dataset), result_acc

    return acc / len(dataloader.dataset)

def generate_metrics(srcnn, classifier, test_batch_size, name, dataset):
    if dataset == "ilsvrc":
        cln_dataloader = data.DataLoader(
            load_imagenet(crpMode="clean", tsfrmMode="eval"),
            batch_size=test_batch_size,
            shuffle=False,
            num_workers=16,
            pin_memory=True,
            drop_last=False,
        )

        tng_dataloader = data.DataLoader(
            load_imagenet(crpMode="train", tsfrmMode="eval"),
            batch_size=test_batch_size,
            shuffle=False,
            num_workers=16,
            pin_memory=True,
            drop_last=False,
        )

        val_dataloader = data.DataLoader(
            load_imagenet(crpMode="val", tsfrmMode="eval"),
            batch_size=test_batch_size,
            shuffle=False,
            num_workers=16,
            pin_memory=True,
            drop_last=False,
        )
    else:
        raise NotImplementedError

    print("... validating cln_dataloader")
    cln_acc = give_me_acc(srcnn, classifier, cln_dataloader)
    print("... validating tng_dataloader")
    tng_acc, tng_acc_dict = give_me_acc(srcnn, classifier, tng_dataloader, plot_acc=True)
    print("... validating val_dataloader")
    val_acc, val_acc_dict = give_me_acc(srcnn, classifier, val_dataloader, plot_acc=True)

    result_dict = {
        "cln_acc": cln_acc,
        "tng_acc": tng_acc,
        "val_acc": val_acc,
    }

    with open(f"saved_models/{name}/best_acc.json", "w") as fp:
        json.dump(result_dict, fp)

wandb.init(project="urie-extension", name=opt.desc, config=opt)
wandb.watch(srcnn)

acc = 0.
best_acc = 0.
for epoch in range(1, opt.epochs + 1):
    train(epoch)
    if epoch % 5 == 0:
        acc = validate(epoch)

    if acc > best_acc and epoch > 10:
        wandb.run.summary["best_accuracy"] = acc
        best_acc = acc
        checkpoint(best_acc, opt.save)
