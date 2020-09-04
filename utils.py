import torch
import wandb

import torch.nn as nn
import numpy as np
import scipy.ndimage as ndimage
import torchvision.utils as vutils

from typing import NamedTuple


from torch.nn.modules.batchnorm import _BatchNorm
from torch.nn.parameter import Parameter
from torch.nn import functional as F
import torch


class _BatchInstanceNorm(_BatchNorm):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True):
        super(_BatchInstanceNorm, self).__init__(num_features, eps, momentum, affine)
        self.gate = Parameter(torch.Tensor(num_features))
        self.gate.data.fill_(1)
        setattr(self.gate, 'bin_gate', True)

    def forward(self, input):
        self._check_input_dim(input)

        # Batch norm
        if self.affine:
            bn_w = self.weight * self.gate
        else:
            bn_w = self.gate
        out_bn = F.batch_norm(
            input, self.running_mean, self.running_var, bn_w, self.bias,
            self.training, self.momentum, self.eps)

        # Instance norm
        b, c  = input.size(0), input.size(1)
        if self.affine:
            in_w = self.weight * (1 - self.gate)
        else:
            in_w = 1 - self.gate
        input = input.view(1, b * c, *input.size()[2:])
        out_in = F.batch_norm(
            input, None, None, None, None,
            True, self.momentum, self.eps)
        out_in = out_in.view(b, c, *input.size()[2:])
        out_in.mul_(in_w[None, :, None, None])

        return out_bn + out_in


class BatchInstanceNorm2d(_BatchInstanceNorm):
    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'.format(input.dim()))

class EnhanceConfig(NamedTuple):
    pass


class ClfConfig(NamedTuple):
    pass

class SobelFilter(nn.Module):
    def __init__(self):
        super(SobelFilter, self).__init__()
        grayscale = nn.Conv2d(3, 1, kernel_size=1, stride=1, padding=0)
        grayscale.weight.data.fill_(1.0 / 3.0)
        grayscale.bias.data.zero_()
        sobel_filter = nn.Conv2d(1, 2, kernel_size=3, stride=1, padding=1)
        sobel_filter.weight.data[0, 0].copy_(
            torch.FloatTensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
        )
        sobel_filter.weight.data[1, 0].copy_(
            torch.FloatTensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
        )
        sobel_filter.bias.data.zero_()
        self.sobel = nn.Sequential(grayscale, sobel_filter)

        for p in self.sobel.parameters():
            p.requires_grad = False
    def forward(self, x):
        return self.sobel(x)


class GaussianLayer(nn.Module):
    def __init__(self):
        super(GaussianLayer, self).__init__()
        self.seq = nn.Sequential(
            nn.ReflectionPad2d(10),
            nn.Conv2d(3, 3, 21, stride=1, padding=0, bias=None, groups=3)
        )

        self.weights_init()
    def forward(self, x):
        return self.seq(x)

    def weights_init(self):
        n= np.zeros((21,21))
        n[10,10] = 1
        k = ndimage.gaussian_filter(n,sigma=3)
        for name, f in self.named_parameters():
            f.data.copy_(torch.from_numpy(k))


def ConvertColor_RGB2YCbCr(image):
    # image should be in shape (B, C, H, W) torch.tensor
    r = image[:, 0, :, :]
    g = image[:, 1, :, :]
    b = image[:, 2, :, :]

    Y = 0.299 * r + 0.587 * g + 0.114 * b
    Cb = 0.564 * (b - Y)
    Cr = 0.731 * (r - Y)

    image = torch.stack([Y, Cb, Cr], dim=1)
    return image


def ConvertColor_YCbCr2RGB(image):
    # image should be in shape (B, C, H, W) torch.tensor
    Y = image[:, 0, :, :]
    Cb = image[:, 1, :, :]
    Cr = image[:, 2, :, :]

    r = Y + 1.402 * Cr
    g = Y - 0.344 * Cb - 0.714 * Cr
    b = Y + 1.772 * Cb

    image = torch.stack([r, g, b], dim=1)
    return image


def plot_images_to_wandb(images: list, name: str):
    # images are should be list of RGB images tensors in shape (C, H, W)
    images = vutils.make_grid(images, normalize=True, range=(-2.11785, 2.64005))

    if images.dim() == 3:
        images = images.permute(1, 2, 0)
    images = images.detach().cpu().numpy()

    images = wandb.Image(images, caption=name)

    wandb.log({name: images})

def plot_network(model, name):
    from torchviz import make_dot
    import torch
    from torch.autograd import Variable

    x = Variable(torch.randn(32, 3, 227, 227))
    y = model(x)
    g = make_dot(y, params=dict(model.named_parameters()))
    g.format = "png"
    g.render(f"model_plots/{name}")
