import torch

from torchvision import models
from torch import nn
from tqdm import tqdm
from sklearn import metrics
import numpy as np

import sys
from os.path import dirname as dir
sys.path.append(dir(sys.path[0]))
from torch.utils.tensorboard import SummaryWriter

from dataset.read_data_list import *
from dataset.prepare_data import *

from model.model_build import MLP, vgg, vit, mobil_v3,resnet
from utils import Const, convert_category_multi, convert_category_index

device = "cuda" if torch.cuda.is_available() else "cpu"

# pathology classification task
#classes = [category['name'] for category in Const.category_dict]
# prompt retrieval task
classes = Const.classes
num_classes = len(classes)

#model = vgg(num_classes).to(device)
model = vit(num_classes).to(device)
#model = mobil_v3(num_classes).to(device)
#model = resnet(num_classes).to(device)


#preprocess = transforms.ToTensor()
preprocess = model.preprocess

train_loader, test_loader = prepare_data(batch_size_test=64, image_preprocessor=preprocess)
#model.load_state_dict(torch.load('./model_weights/vgg_best.pth'))
model.load_state_dict(torch.load('./model_weights/vit_best.pth'))
#model.load_state_dict(torch.load('./model_weights/mobil_best.pth'))
#model.load_state_dict(torch.load('./model_weights/resnet_best.pth'))



model.eval()

def predict(logits):
    sigmoid = nn.Sigmoid()
    prob = sigmoid(logits)
    predicts = (prob>0.5).int()
    return predicts

with torch.no_grad():
    logits_bank = np.empty([0,21])
    target_bank = np.array([])
    #accurancy_score = 0
    for data in tqdm(test_loader):
        images, texts = data
        #target = convert_category_index(texts)
        target = np.array([Const.test_classes.index(target) for target in texts])
        images = images.to(device)
        image_features = model(images)
        predicts = torch.argmax(image_features, dim=-1)
        #report = metrics.classification_report(target, predicts.detach().cpu().numpy(), output_dict=False, target_names=classes, digits=4)
        image_features = image_features[:,:21]
        image_features = image_features.softmax(dim=-1).detach().cpu().numpy()
        logits_bank = np.append(logits_bank, image_features,axis=0)
        target_bank = np.append(target_bank, target)
    score_5 = metrics.top_k_accuracy_score(target_bank, logits_bank, k=5)
    score_1 = metrics.top_k_accuracy_score(target_bank, logits_bank, k=1)
    #accurancy_score = accurancy_score/len(test_loader)
    #print(accurancy_score.item())
    print(score_1)
    print(score_5)
    #print(report)
