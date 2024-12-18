from .Const import Const

def convert_category_multi(labels):
  multi_labels = []
  for label in labels:
    abnormalities = label.split(', ')
    multi_label=[0] * len(Const.category_dict)
    for abnormalty in abnormalities:
      for category in Const.category_dict:
        if abnormalty in category['abnormalities']:
          multi_label[category['id']] = 1
    multi_labels.append(multi_label) 
  return multi_labels

def convert_category_index(labels):
  labels_index = []
  for label in labels:
    abnormalities = label.split(', ')
    for abnormalty in abnormalities:
      for category in Const.category_dict:
        if abnormalty in category['abnormalities']:
          label_index = category['id']
    labels_index.append(label_index) 
  return labels_index

def convert_category(labels):
  categories = []
  for label in labels:
    abnormalities = label.split(', ')
    for abnormalty in abnormalities:
      for category in Const.category_dict:
        if abnormalty in category['abnormalities']:
          name = category['name']
    categories.append(name) 
  return categories