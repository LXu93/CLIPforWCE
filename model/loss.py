import torch.nn as nn
import torch

class Contrastive_loss(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device
        self.cross_entropy = nn.CrossEntropyLoss()
    
    def forward(self, logits_image, logits_text):
        ground_truth = torch.arange(len(logits_image),dtype=torch.long, device=self.device)
        total_loss = (self.cross_entropy(logits_image,ground_truth) + self.cross_entropy(logits_text,ground_truth))/2
        #total_loss = self.cross_entropy(logits_image,ground_truth)
        return total_loss


def get_ground_truth_mat(labels):
    mat = torch.zeros([len(labels),len(labels)])
    for i in range(len(labels)):
        for j in range(len(labels)):
            #mat[i,j] = difflib.SequenceMatcher(None,labels[i],labels[j]).ratio()
            if labels[i] == labels[j]:
                mat[i,j] = 1
    return mat

class Contrastive_loss_kl(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device
        self.kl_loss = nn.KLDivLoss(reduction="batchmean")
    
    def forward(self, similarity, ground_truth):
        ground_truth_matrix = get_ground_truth_mat(ground_truth).to(self.device)
        image_similarity = similarity.log_softmax(dim=-1).float()
        text_similarity = image_similarity.T
        text_similarity = similarity.log_softmax(dim=-1).float()
        loss = (self.kl_loss(image_similarity, ground_truth_matrix.softmax(dim=-1)) + self.kl_loss(text_similarity, ground_truth_matrix.softmax(dim=-1)))/2
        return loss