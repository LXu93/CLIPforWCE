import torch
import torch.nn as nn
from torchvision import models


class MLP(nn.Module):
    def __init__(self, input_size, num_classes):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, num_classes)
        #self.fc2 = nn.Linear(input_size//2, num_classes)
        #self.fc3 = nn.Linear(input_size//4, num_classes)
        #self.active = nn.ReLU()
        #self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = x.to(self.fc1.weight.dtype)
        #x = self.dropout(x)
        x = self.fc1(x)
        #x = self.active(x)
        #x = self.dropout(x)
        #x = self.fc2(x)
        #x = self.active(x)
        #x = self.dropout(x)
        #x = self.fc3(x)
        return x
    
class vgg(nn.Module):
    def __init__(self, num_classes, fine_tune = True):
        super().__init__()
        weights = models.VGG19_Weights.DEFAULT
        self.vgg16 = models.vgg19(weights=weights)
        self.preprocess = weights.transforms()

        last_layers = list(self.vgg16.classifier.children())
        last_layers.pop()
        new_last_layers = torch.nn.Sequential(*last_layers)
        self.vgg16.classifier = new_last_layers

        if not fine_tune:
            for param in self.vgg16.parameters():
                param.requires_grad = False

        self.mlp = MLP(4096, num_classes)

    def forward(self, x):
        x = self.vgg16(x)
        x = self.mlp(x)
        return x

class vit(nn.Module):
    def __init__(self, num_classes, fine_tune = True):
        super().__init__()
        weights = models.ViT_L_16_Weights.DEFAULT
        self.vit_model = models.vit_l_16(weights=weights, progress=True)
        self.preprocess = weights.transforms()

        self.vit_model.heads = torch.nn.Identity()

        if not fine_tune:
            for param in self.vit_model.parameters():
                param.requires_grad = False

        self.mlp = MLP(1024, num_classes)

    def forward(self, x):
        x = self.vit_model(x)
        x = self.mlp(x)
        return x
    
class mobil_v3(nn.Module):
    def __init__(self, num_classes, fine_tune = True):
        super().__init__()
        self.mobil = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT, progress=True)

        last_layers = list(self.mobil.classifier.children())
        last_layers.pop()
        new_last_layers = torch.nn.Sequential(*last_layers)
        self.mobil.classifier = new_last_layers

        if not fine_tune:
            for param in self.mobil.parameters():
                param.requires_grad = False

        self.mlp = MLP(1280, num_classes)
    
    def forward(self, x):
        x = self.mobil(x)
        x = self.mlp(x)
        return x

class resnet(nn.Module):
    def __init__(self, num_classes, fine_tune = True):
        super().__init__()
        self.resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT, progress=True)
        self.resnet.fc = torch.nn.Identity()

        if not fine_tune:
            for param in self.resnet.parameters():
                param.requires_grad = False

        self.mlp = MLP(2048, num_classes)
    
    def forward(self, x):
        x = self.resnet(x)
        x = self.mlp(x)
        return x