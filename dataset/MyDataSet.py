import os

import torch
from PIL import Image

class MyDataSet(torch.utils.data.Dataset):
  def __init__(self, image_dir, image_path_list, text_list, transform=None):
    super().__init__()
    self.image_dir = image_dir
    self.image_path_list = image_path_list
    self.text_list = text_list
    self.transform = transform

  def __len__(self):
    return len(self.image_path_list)

  def __getitem__(self, idx):
    image_path = os.path.join(self.image_dir, self.image_path_list[idx])
    image = Image.open(image_path)
    text = self.text_list[idx]
    if self.transform:
      image = self.transform(image)
    return image, text