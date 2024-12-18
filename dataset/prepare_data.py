import torch
import copy

from torchvision import transforms
from torch.utils.data import DataLoader, Sampler
from collections import defaultdict
from torch import nn
from tqdm import tqdm

import random
import sys
from os.path import dirname as dir
sys.path.append(dir(sys.path[0]))
from torch.utils.tensorboard import SummaryWriter

from .MyDataSet import MyDataSet
from .read_data_list import *
from utils import generate_random_prompt, convert_to_multilabel

class DiverseClassSampler(Sampler):
    def __init__(self, data_source, batch_size):
        self.data_source = data_source
        self.batch_size = batch_size
        self.original_class_to_indices = self._group_by_class()  # Store the original mapping
        self.class_to_indices = copy.deepcopy(self.original_class_to_indices)  # Copy for each epoch

    def _group_by_class(self):
        # Group indices by their class labels
        class_to_indices = defaultdict(list)
        for idx, (_, label) in enumerate(self.data_source):
            class_to_indices[label].append(idx)
        return class_to_indices

    def __iter__(self):
        # Reset the class_to_indices at the start of each epoch
        self.class_to_indices = copy.deepcopy(self.original_class_to_indices)
        indices = []
        class_pool = list(self.class_to_indices.keys())

        # While there are still classes with samples
        while any(self.class_to_indices.values()):
            batch_indices = []
            selected_classes = set()

            # Try to fill the batch with unique classes
            for cls in class_pool:
                # Break if the batch is full
                if len(batch_indices) >= self.batch_size:
                    break
                # Check if the class has samples left and hasn't been selected in this batch
                if cls not in selected_classes and self.class_to_indices[cls]:
                    batch_indices.append(self.class_to_indices[cls].pop())
                    selected_classes.add(cls)

            # If there are not enough unique classes to fill the batch, allow repetitions
            remaining_slots = self.batch_size - len(batch_indices)
            if remaining_slots > 0:
                # Refill from any available classes (including duplicates if necessary)
                available_indices = [self.class_to_indices[cls].pop() 
                                     for cls in class_pool 
                                     if self.class_to_indices[cls]]
                batch_indices.extend(available_indices[:remaining_slots])

            # Clean up classes that are now empty
            class_pool = [cls for cls in class_pool if self.class_to_indices[cls]]
            indices.extend(batch_indices)

        return iter(indices)

    def __len__(self):
        return len(self.data_source)

def randomly_select_images(image_paths, labels, n=1):

    label_to_images = {}
    for i, label in enumerate(labels):
        if label not in label_to_images:
            label_to_images[label] = []
        label_to_images[label].append(image_paths[i])
    
    selected_images = []
    selected_labels = []

    for label, image_list in label_to_images.items():
        num_images = len(image_list)

        if num_images > n:
            selected_indices = random.sample(range(num_images), n)
            for index in selected_indices:
                selected_images.append(image_list[index])
                selected_labels.append(label)
        elif num_images > 0:
            selected_images.extend(image_list)
            selected_labels.extend([label] * num_images)
    
    return selected_images, selected_labels
       
def prepare_data(
        extended = False,
        batch_size_train=64,
        batch_size_test = 64,
        image_preprocessor=None, 
        text_preprocessor=None,
        few_shot = False,
        n_shots =1,
        shuffle=True):
    if not extended:
        data_path = './data/CAPTIV8_meta_frame_video_labeled_split.xlsx'
        columns = [1, 4, 8, 9]

        image_path_train, label_train = read_excel_sheets(data_path, columns=columns,split='train', text_preprocess=text_preprocessor)
        image_path_test, label_test = read_excel_sheets(data_path, columns=columns,split='test',text_preprocess=text_preprocessor)

        train_image_dir = './data/Frames'
        test_image_dir = './data/Frames'
        
    else:
        data_path = './data/extended_image_list_lpips_labeled.xlsx'
        columns = [4,6,7,8]
        image_path_train, label_train = read_extended_excel(data_path, columns=columns,split='train', text_preprocess=text_preprocessor)
        
        test_data_path = './data/CAPTIV8_meta_frame_video_labeled_split.xlsx'
        test_columns = [1, 4, 8, 9]
        image_path_test, label_test = read_excel_sheets(test_data_path, columns=test_columns,split='test',text_preprocess=text_preprocessor)

        train_image_dir = './data'
        test_image_dir = './data/Frames'

    #classes = list(set(label_train+label_test))
    transform = transforms.Compose([transforms.CenterCrop(size=512),
                                    image_preprocessor])
    
    if few_shot:
        image_path_train, label_train = randomly_select_images(image_path_train,label_train,n_shots)

    # Load the dataset
    train_set = MyDataSet(train_image_dir, image_path_train, label_train, transform=transform)
    test_set = MyDataSet(test_image_dir, image_path_test, label_test, transform=transform)
    
    if not few_shot:
        sampler =DiverseClassSampler(train_set, batch_size_train)
        train_loader = DataLoader(train_set, batch_size=batch_size_train, drop_last=True, shuffle=shuffle)
    else:    
        train_loader = DataLoader(train_set, batch_size=batch_size_train,shuffle=True)
    
    if batch_size_test == 'full':
        batch_size_test = len(test_set)
    test_loader = DataLoader(test_set, batch_size=batch_size_test)
    return train_loader, test_loader

if __name__ == '__main__':
    train_loader, test_loader = prepare_data(extended=True, batch_size_test='full', text_preprocessor = None, few_shot=True)
    for batch in train_loader:
        print(batch[0])
