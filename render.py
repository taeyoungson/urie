import sys
import torch
import torchvision.utils as vutils
import torchvision.transforms as transforms

from PIL import Image
from pathlib import Path
from models.skunet_model import SKUNet

def main():
    pretrained_weights = "./ECCV_MODELS/ECCV_SKUNET_OURS.ckpt.pt"
    image = Image.open(sys.argv[1])
    urie = SKUNet().cuda().eval()

    weights = torch.load(pretrained_weights)

    # model weights are trained from multi-gpu format
    new_weights = {}
    for k, v in weights.items():
        if "module." in k:
            new_weights[k.replace("module.", "")] = v
        else:
            new_weights[k] = v
    urie.load_state_dict(new_weights, strict=True)

    tsfrms = transforms.Compose([
        transforms.Resize((227, 227)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = tsfrms(image).unsqueeze(0).cuda()
    output_image, _ = urie(image)
    vutils.save_image(torch.cat((image, output_image), dim=0), "./output.jpg", normalize=True, scale_each=True)

if __name__ == '__main__':
    main()
