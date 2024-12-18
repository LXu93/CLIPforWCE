import torch

from torchvision import models
from torch import nn
from tqdm import tqdm

import sys
from os.path import dirname as dir
sys.path.append(dir(sys.path[0]))
from torch.utils.tensorboard import SummaryWriter

from dataset.read_data_list import *
from dataset.prepare_data import *

from model.model_build import MLP, vgg, vit, mobil_v3, resnet
from utils import Const, convert_category_multi, convert_category_index

device = "cuda" if torch.cuda.is_available() else "cpu"

classes = [category['name'] for category in Const.category_dict]
#classes = Const.classes
output_size = len(classes)

num_classes = len(classes)

model = vit(num_classes, fine_tune=True).to(device)
#model = vgg(num_classes,fine_tune=True).to(device)
#model = mobil_v3(num_classes, fine_tune=True).to(device)
#model = resnet(num_classes,fine_tune=True).to(device)


preprocess = model.preprocess
#preprocess = transforms.ToTensor()

train_loader, test_loader = prepare_data(extended=False, 
                                         batch_size_train=32, 
                                         image_preprocessor=preprocess,
                                         few_shot=False,
                                         n_shots=4)

weights = torch.Tensor([ 0.87172012,  0.2232454 ,  1.25630252,  4.27142857,  5.33928571, 11.64935065, 14.23809524]).to(device)
#pos_weights = torch.Tensor([ 5.1020,  0.5627,  7.0811, 28.9000, 36.3750, 80.5455, 98.6667]).to(device)
#pos_weights = torch.Tensor([ 1.7988,   1.3481,  68.5385,  10.8947,   7.9505,  81.1818, 2.0748,   3.9945,  36.6667, 128.1429,  29.1333]).to(device)
#criterion =nn.BCEWithLogitsLoss(pos_weight=pos_weights)
criterion = nn.CrossEntropyLoss(weight=weights)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

num_epochs = 200


for epoch in range(num_epochs):
    model.train()
    mean_train_loss = 0.0
    pbar = tqdm(train_loader)
    for batch in pbar:
        images, texts = batch
        labels = convert_category_index(texts)
        #labels = [classes.index(target) for target in texts]
        labels = torch.tensor(labels,dtype=torch.long, device=device)
        images = images.to(device)
        output = model(images)
        loss = criterion(output, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        mean_train_loss += loss.item()
        pbar.set_description(f"Epoch {epoch+1}/{num_epochs}, Loss: {mean_train_loss/len(train_loader):.4f}")
    
    
    model.eval()
    best_loss = 10
    if (epoch+1)%1 == 0:
        test_loss = 0.0
        with torch.no_grad():
            for data in test_loader:
                test_images, test_texts = data
                test_images= test_images.to(device)
                test_labels = convert_category_index(test_texts)
                #test_labels = [classes.index(target) for target in test_texts]
                test_labels = torch.tensor(test_labels,dtype=torch.long, device=device)
        
                pre = model(test_images)
                
                batch_error = criterion(pre, test_labels)
                test_loss = test_loss + batch_error.item()
        test_loss = test_loss/len(test_loader)
        print(f'Test Loss: {test_loss}')
        if test_loss<best_loss:
            best_loss = test_loss
            #torch.save(model.state_dict(), f'./model_weights/vgg_best.pth')
            torch.save(model.state_dict(), f'./model_weights/vit_best.pth')
            #torch.save(model.state_dict(), f'./model_weights/mobil_best.pth')
            #torch.save(model.state_dict(), f'./model_weights/resnet_best.pth')


    if (epoch+1)%5 == 6:
        #torch.save(model.state_dict(), f'./model_weights/vgg_{epoch+1}.pth')
        torch.save(model.state_dict(), f'./model_weights/vit_{epoch+1}.pth')
        #torch.save(model.state_dict(), f'./model_weights/mobil_{epoch+1}.pth')
        #torch.save(model.state_dict(), f'./model_weights/resnet_{epoch+1}.pth')

