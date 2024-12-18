import clip
import torch

import torch.nn.functional as F

import sys
from os.path import dirname as dir
sys.path.append(dir(sys.path[0]))
from torch.utils.tensorboard import SummaryWriter


from model.model_build import MLP, vgg, vit, mobil_v3, resnet


device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load('ViT-L/14', device)
#model = vit(num_classes, fine_tune=True).to(device)

total_params = sum(p.numel() for p in model.parameters())
print(f"Number of parameters: {total_params}")
