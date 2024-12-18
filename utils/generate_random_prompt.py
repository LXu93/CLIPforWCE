from random import randrange
from .Const import Const
def generate_random_prompt(multi_labels):
  prompts = []
  random_range = len(Const.abnormality_list)
  resample_times = 5
  for multi_label in multi_labels:
    idx = randrange(random_range)
    if multi_label[idx]==0:
      exclude = [idx]
      for rep in range(1,resample_times):
        new_idx = randrange(random_range)
        while new_idx in exclude:
          new_idx = randrange(random_range)
        if multi_label[new_idx]==0:
          exclude.append(new_idx)
        else:
          prompts.append(f'endoscopy image with {Const.abnormality_list[new_idx]}')
          break
      if len(exclude) == resample_times:
          prompts.append(f'endoscopy image not with {Const.abnormality_list[new_idx]}')
    else:
      prompts.append(f'endoscopy image with {Const.abnormality_list[idx]}')
  return prompts