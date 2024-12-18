from .Const import Const

def convert_to_multilabel(labels):
  multilabels = []
  for label in labels:
    abnormalities = label.split(', ')
    multilabel = []
    for abnormality in Const.abnormality_list:
      if abnormality in abnormalities:
        multilabel.append(1)
      else:
        multilabel.append(0)
    multilabels.append(multilabel)
  return multilabels

def convert_to_multilabel(labels):
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