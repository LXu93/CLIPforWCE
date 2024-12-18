import os
import re
import pandas as pd
import numpy as np

import sys
from os.path import dirname as dir
sys.path.append(dir(sys.path[0]))
from utils import generate_random_prompt, convert_to_multilabel, Const

def read_excel_sheets(file_path, columns=None, sheets=None, split='train', text_preprocess=None):
    excel_data = pd.read_excel(file_path, sheet_name=sheets, usecols=columns)
    combined_df = pd.concat(excel_data.values(), ignore_index=True)
    combined_df = combined_df[combined_df['split']==split]
    image_path_list = []
    for id, image_path in zip(combined_df['Patient ID'],combined_df['image_path']):
        image_path_list.append(os.path.join(id,image_path))
    #text_list = combined_df['image_label'].tolist()
    label_list = combined_df['refined_label'].tolist()
    if text_preprocess:
        #text_list = text_preprocess(text_list)
        label_list = text_preprocess(label_list)

    return image_path_list, label_list

def generate_all_prompt(labels):
  all_prompts = []
  for label in labels:
    abnormalities = label.split(', ')
    for abnormality in Const.abnormality_list:
      if abnormality in abnormalities:
        all_prompts.append(f'endoscopy image with {abnormality}')
      else:
        all_prompts.append(f'endoscopy image without {abnormality}')
  return all_prompts

def read_and_generate_prompt(file_path, columns=None, sheets=None, split='train'):
    excel_data = pd.read_excel(file_path, sheet_name=sheets, usecols=columns)
    combined_df = pd.concat(excel_data.values(), ignore_index=True)
    combined_df = combined_df[combined_df['split']==split]
    image_path_list = []
    for id, image_path in zip(combined_df['Patient ID'],combined_df['image_path']):
        image_path_list.append(os.path.join(id,image_path))
    expand_image = np.repeat(np.array(image_path_list), len(Const.abnormality_list), axis=0)
    all_prompts = generate_all_prompt(combined_df['refined_label'])

    return expand_image, all_prompts

def remove_extra_space(text_list):
    for idx in range(len(text_list)):
        text_list[idx] = text_list[idx].strip()
        text_list[idx] = re.sub('\s\s+', ' ', text_list[idx])
        text_list[idx] = text_list[idx].replace(".",",")
        text_list[idx] = text_list[idx].replace(" ,",",")
        if text_list[idx][-1] == ",":
          text_list[idx] = text_list[idx][:-1]
    return text_list

def text_template(text_list):
    template = 'endoscopy image with '
    for idx in range(len(text_list)):
        text_list[idx] = template + text_list[idx]
    return text_list

def compress_labels(text_list):
    text_list = remove_extra_space(text_list)
    for idx in range(len(text_list)):
        abnormalities = []
        for abnormality in Const.abnormality_list:
            if abnormality in text_list[idx]:
              if 'pseudo'+abnormality not in text_list[idx]:
               pattern = rf'no([^,]*){abnormality}'
               # Use re.search to find the pattern
               negative = re.search(pattern, text_list[idx])
               if not negative:
                    abnormalities.append(abnormality)
        if not abnormalities:
            abnormalities.append('normal')
        text_list[idx] = ', '.join(abnormalities)
    return text_list

def read_extended_excel(file_path, columns=None, split = 'train', text_preprocess=None):
    df = pd.read_excel(file_path, usecols=columns)
    selected_df = df[(df['loss'] < 0.09) & (df['split'] == split)]
    image_path_list = selected_df['path'].tolist()
    text_list = selected_df['label'].tolist()
    if text_preprocess:
        text_list = text_preprocess(text_list)
    return image_path_list, text_list

