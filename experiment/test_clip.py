import clip
import torch
from sklearn.metrics import top_k_accuracy_score, classification_report
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

import sys
from os.path import dirname as dir
sys.path.append(dir(sys.path[0]))

from dataset.prepare_data import *
from torch.utils.tensorboard import SummaryWriter
from utils import Const, convert_category_index


classes = Const.test_classes
#classes = [category['name'] for category in Const.category_dict]

# Load the model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load('ViT-B/16', device)
# Prepare the inputs

train_loader, test_loader = prepare_data(extended=False, 
                                         batch_size_test='full', 
                                         batch_size_train= 32, 
                                         image_preprocessor = preprocess, 
                                         text_preprocessor = None,
                                         shuffle=False
                                         )

#writer = SummaryWriter()
for i in range(1,2):
    model.load_state_dict(torch.load(f'./model_weights/finetuned_clip_kl_best.pth'))
    #model.load_state_dict(torch.load(f'./model_weights/finetuned_clip_multi_extended_{i*5}.pth'))


    model.eval()
    score_5 = 0
    score_1 = 0
    logits_bank = np.empty([0,21])
    target_bank = np.array([])
    with torch.no_grad():
        for batch in tqdm(test_loader):
            inputs, ground_truth = batch
            # caption retrieval ground truth
            ground_truth_index = np.array([classes.index(target) for target in ground_truth])
            # classification ground truth
            #ground_truth_index = convert_category_index(ground_truth)            
            inputs = inputs.to(device)
            image_features = model.encode_image(inputs)
            text_inputs = torch.cat([clip.tokenize(f"endoscopy image with {label}") for label in classes]).to(device)
            text_features = model.encode_text(text_inputs)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            similarity = (image_features @ text_features.T).softmax(dim=-1)
            #predicts = torch.argmax(similarity, dim=-1)
            #report = classification_report(ground_truth_index, predicts.detach().cpu().numpy(), output_dict=False, target_names=classes, digits=4)
            # replace criterion
            similarity = similarity.detach().cpu().numpy()
            logits_bank = np.append(logits_bank, similarity,axis=0)
            target_bank = np.append(target_bank, ground_truth_index)
        score_5 = top_k_accuracy_score(target_bank, logits_bank, k=5)
        score_1 = top_k_accuracy_score(target_bank, logits_bank, k=1)
    #print(f'at {i*5} epoch')
    print(f'top 1: {100*score_1:0.2f}%')
    print(f'top 5: {100*score_5:0.2f}%')
    #print(report)
    #writer.add_scalar('top 5 accuracy', score_5/len(test_loader),i*5)
    #writer.add_scalar('top 1 accuracy', score_1/len(test_loader),i*5)
    #writer.flush()