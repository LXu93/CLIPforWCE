import clip
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm


import sys
from os.path import dirname as dir
sys.path.append(dir(sys.path[0]))

from dataset.prepare_data import *
from model.model_build import MLP
from dataset.MyDataSet import MyDataSet
from dataset.read_data_list import *
from utils import Const, convert_category_multi, convert_category_index

# Load the model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load('ViT-L/14@336px', device)
#model.load_state_dict(torch.load('./model_weights/finetuned_clip_best.pth'))
model.load_state_dict(torch.load('./model_weights/finetuned_clip_best.pth'))

#model.load_state_dict(torch.load('./model_weights/finetuned_clip_multi_60.pth'))
#model.load_state_dict(torch.load('./model_weights/finetuned_clip_multi_extended_80.pth'))
#model.load_state_dict(torch.load('./model_weights/finetuned_clip_all_5.pth'))
#model.load_state_dict(torch.load('./model_weights/finetuned_clip_negative_90.pth'))
#model.load_state_dict(torch.load('./model_weights/finetuned_clip_extended_35.pth'))
model.eval()  # model in train mode by default, impacts some models with BatchNorm or stochastic depth active

train_loader, test_loader= prepare_data(extended=False, image_preprocessor=preprocess)

classes = [category['name'] for category in Const.category_dict]
#classes = Const.classes


input_size = 512
input_size = 768
output_size = len(classes)

mlp = MLP(input_size, output_size).to(device)

weights = torch.Tensor([ 0.87172012,  0.2232454 ,  1.25630252,  4.27142857,  5.33928571, 11.64935065, 14.23809524]).to(device)
criterion = nn.CrossEntropyLoss(weight=weights)
optimizer = torch.optim.Adam(mlp.parameters(), lr=1e-4)

num_epochs = 100

for epoch in range(num_epochs):
    mlp.train()
    train_loss = 0.0
    pbar = tqdm(train_loader)
    for batch in pbar:
        images, texts = batch
        #labels = convert_category_multi(texts)
        labels = convert_category_index(texts)
        #labels = [classes.index(target) for target in texts]
        labels = torch.tensor(labels,dtype=torch.long, device=device)
        images = images.to(device)
        with torch.no_grad():
            image_features = model.encode_image(images)
        output = mlp(image_features)
        loss = criterion(output, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss
        pbar.set_description(f"Epoch {epoch+1}/{num_epochs}, Loss: {train_loss.item():.4f}")
    
    
    mlp.eval()
    best_loss = 100
    if (epoch+1)%1 == 0:
        test_loss = 0.0
        with torch.no_grad():
            for data in test_loader:
                test_images, test_texts = data
                test_images= test_images.to(device)
                test_labels = convert_category_index(test_texts)
                #test_labels = [classes.index(target) for target in test_texts]
                test_labels = torch.tensor(test_labels,dtype=torch.long, device=device)
                
                test_images_features = model.encode_image(test_images)
                pre = mlp(test_images_features)
                
                pre_error = criterion(pre, test_labels)
                test_loss = test_loss + pre_error.item()
        test_loss = test_loss/len(test_loader)
        print(f'Test Loss: {test_loss}')
        if test_loss<best_loss:
            torch.save(mlp.state_dict(), f'./model_weights/mlp_best.pth')
    if (epoch+1)%5 == 0:
        torch.save(mlp.state_dict(), f'./model_weights/mlp_{epoch+1}.pth')
