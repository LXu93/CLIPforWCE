import clip
import torch

from torch import nn
from tqdm import tqdm
import torch.nn.functional as F
from sklearn.metrics import top_k_accuracy_score

import sys
from os.path import dirname as dir
sys.path.append(dir(sys.path[0]))
from torch.utils.tensorboard import SummaryWriter

from dataset.read_data_list import *
from dataset.prepare_data import *
from utils import generate_random_prompt, convert_to_multilabel, Const, convert_category,convert_category_index
from model.loss import Contrastive_loss, Contrastive_loss_kl
# Load the model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load('ViT-B/16', device)

lr=1e-6
weight_decay = 0
batch_size = 8

# prompt retrieval task
classes =  Const.test_classes
# pathology classification task
#classes = [category['name'] for category in Const.category_dict]
num_classes = len(classes)

train_loader, test_loader = prepare_data(extended=False, 
                                         batch_size_test='full', 
                                         batch_size_train= batch_size, 
                                         image_preprocessor = preprocess, 
                                         text_preprocessor = None, 
                                         few_shot=False, 
                                         n_shots=4)

optimizer = torch.optim.Adam(model.parameters(), lr=lr,betas=(0.9,0.98),eps=1e-6,weight_decay=weight_decay)
# Original loss
criterion = Contrastive_loss(device)
# KL loss
#criterion = Contrastive_loss_kl(device)

num_epochs = 100
writer = SummaryWriter()
writer.add_scalar('hyperparameter/learning rate',lr)
writer.add_scalar('hyperparameter/weight decay', weight_decay)
writer.add_scalar('hyperparameter/batch_size', batch_size)
writer.flush()

best_acc1 = 0
for epoch in range(num_epochs):
    pbar = tqdm(train_loader)
    model.train()
    mean_loss = 0
    for batch in pbar:

        images,texts = batch 
        
        images= images.to(device)
        #texts = convert_category(texts) # category labels
        text_tokens = torch.cat([clip.tokenize(f"endoscopy image with {label}") for label in texts]).to(device).to(device)

        # Forward pass
        logits_image, logits_text = model(images, text_tokens)
        # KL_loss
        #total_loss = criterion(logits_image, texts)
        # Original loss
        total_loss = criterion(logits_image, logits_text)
      
        # Backward pass
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        mean_loss = mean_loss + total_loss.item()
        pbar.set_description(f"Epoch {epoch+1}/{num_epochs}, Loss: {mean_loss/len(train_loader):.4f}")
    mean_loss = mean_loss/len(train_loader)
    writer.add_scalar("Loss/Train Loss", mean_loss, epoch)
    writer.flush()
    if (epoch+1)%10 == 0:
        torch.save(model.state_dict(),f'./model_weights/finetuned_clip_wd_{epoch+1}.pth')

    model.eval()
    if (epoch+1)%1 == 0:
        test_loss = 0
        with torch.no_grad():
            for data in test_loader:
                test_images, test_texts = data
                test_images= test_images.to(device)
                #test_texts = convert_category(test_texts) #category label
                test_text_tokens = torch.cat([clip.tokenize(f"endoscopy image with {label}") for label in test_texts]).to(device).to(device)
                test_image_features = model.encode_image(test_images)
                test_text_features = model.encode_text(test_text_tokens)
                test_image_features =F.normalize(test_image_features,dim=-1)
                test_text_features = F.normalize(test_text_features,dim=-1)
                test_logits_image = torch.matmul(test_image_features, test_text_features.T)
                test_logits_text = test_logits_image.T
                # KL-loss
                #loss = criterion(test_logits_image, test_texts)
                # Original loss
                loss = criterion(test_logits_image, test_logits_text)
                test_loss = test_loss + loss.item()

        print(f'Test Loss: {test_loss/len(test_loader)}')
        writer.add_scalar("Loss/Test Loss", test_loss/len(test_loader), epoch)
        writer.flush()
        score_5 = 0
        score_1 = 0
        logits_bank = np.empty([0,num_classes])
        target_bank = np.array([])
        with torch.no_grad():
            for batch in test_loader:
                inputs, ground_truth = batch
                #ground_truth_index = convert_category_index(ground_truth) # category labels
                ground_truth_index = np.array([classes.index(target) for target in ground_truth])
                inputs = inputs.to(device)
                image_features = model.encode_image(inputs)
                text_inputs = torch.cat([clip.tokenize(f"endoscopy image with {label}") for label in classes]).to(device)
                text_features = model.encode_text(text_inputs)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                text_features /= text_features.norm(dim=-1, keepdim=True)
                similarity = (image_features @ text_features.T).softmax(dim=-1)
                similarity = similarity.detach().cpu().numpy()
                logits_bank = np.append(logits_bank, similarity,axis=0)
                target_bank = np.append(target_bank, ground_truth_index)
        score_5 = top_k_accuracy_score(target_bank, logits_bank, k=5)
        score_1 = top_k_accuracy_score(target_bank, logits_bank, k=1)
        print(f'top 1: {100*score_1:0.2f}%')
        print(f'top 5: {100*score_5:0.2f}%')
        writer.add_scalar('Acc/top 5 accuracy', score_5,epoch)
        writer.add_scalar('Acc/top 1 accuracy', score_1,epoch)
        writer.flush()
        if score_1 > best_acc1:
            torch.save(model.state_dict(),f'./model_weights/finetuned_clip_wd_best.pth')
            best_acc1 = score_1

        
