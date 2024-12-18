import clip
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn import metrics
from torchvision import transforms

import sys
from os.path import dirname as dir
sys.path.append(dir(sys.path[0]))

from model.model_build import MLP
from dataset.prepare_data import *
from utils import Const, convert_to_multilabel, convert_category_multi, convert_category_index

classes = [category['name'] for category in Const.category_dict]
#classes = Const.classes

# Load the model
device = "cuda" if torch.cuda.is_available() else "cpu"

clip_model, preprocess = clip.load('ViT-L/14@336px', device)
clip_model.load_state_dict(torch.load('./model_weights/finetuned_clip_best.pth'))

train_loader, test_loader = prepare_data(batch_size_test='full', image_preprocessor=preprocess)
#clip_model.load_state_dict(torch.load('./model_weights/finetuned_clip_extended_35.pth'))
clip_model.eval()  # model in train mode by default, impacts some models with BatchNorm or stochastic depth active

#input_size = 512
input_size =768
output_size = len(classes)

mlp = MLP(input_size, output_size).to(device)
mlp.load_state_dict(torch.load('./model_weights/mlp_best.pth'))
mlp.eval()

def predict(logits):
    sigmoid = nn.Sigmoid()
    prob = sigmoid(logits)
    predicts = (prob>0.50).int()
    return predicts
with torch.no_grad():
    #accurancy_score = 0
    for data in test_loader:
        images, texts = data
        target = convert_category_index(texts)
        #target = np.array([Const.test_classes.index(target) for target in texts])
        images = images.to(device)
        image_features = clip_model.encode_image(images)
        #predicts = predict(mlp(image_features))
        #accurancy = 1 - torch.count_nonzero(predicts-target)/torch.numel(target)
        #accurancy_score += accurancy
        logits = mlp(image_features)
        predicts = torch.argmax(logits, dim=-1)
        report = metrics.classification_report(target, predicts.detach().cpu().numpy(), output_dict=False, target_names=classes, digits=4)
        #image_features = logits[:,:21]
        #image_features = image_features.softmax(dim=-1).detach().cpu().numpy()
        #score_5 = metrics.top_k_accuracy_score(target, image_features, k=5)
        #score_1 = metrics.top_k_accuracy_score(target, image_features, k=1)
    #accurancy_score = accurancy_score/len(test_loader)
    #print(accurancy_score.item())
    print(report)
    #print(score_1)
    #print(score_5)
