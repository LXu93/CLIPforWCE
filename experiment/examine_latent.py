import clip
import torch
from tqdm import tqdm
from tqdm import tqdm


import sys
from os.path import dirname as dir
sys.path.append(dir(sys.path[0]))

from dataset.prepare_data import *
from utils import Const, convert_category
from utils.dim_reduce import *
from model.model_build import MLP, vgg, vit, mobil_v3,resnet

classes = Const.classes
classes = [category['name'] for category in Const.category_dict]
num_classes = len(classes)


# Load the model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load('ViT-L/14@336px', device)

#model = vit(num_classes, fine_tune=True).to(device)
#model = vgg(num_classes,fine_tune=True).to(device)
#model = mobil_v3(num_classes, fine_tune=True).to(device)
#model = resnet(num_classes,fine_tune=True).to(device)


#preprocess = model.preprocess
#preprocess = transforms.ToTensor()


train_loader, test_loader = prepare_data(extended=False, 
                                         batch_size_test='full', 
                                         batch_size_train= 32, 
                                         image_preprocessor = preprocess, 
                                         text_preprocessor = None,
                                         shuffle=False
                                         )


input_size = 768
#input_size = 1024
#writer = SummaryWriter()

model.load_state_dict(torch.load(f'./model_weights/finetuned_clip_cate_best.pth'))
#model.load_state_dict(torch.load('./model_weights/vit_best.pth'))
model.eval()
with torch.no_grad():
    image_features_bank =np.empty([0,input_size])
    labels_bank = np.array([])
    for data in tqdm(train_loader):
        images, texts = data
        images = images.to(device)
        labels = convert_category(texts)
        #labels = texts
        image_features = model.encode_image(images).detach().cpu().numpy()
        #image_features = model.vit_model(images).detach().cpu().numpy()
        image_features_bank = np.append(image_features_bank, image_features,axis=0)
        labels_bank = np.append(labels_bank,labels)
    umap_embed(image_features_bank, labels_bank, 'Image Embeddings of Fine-tuned CLIP (KL Loss)')
