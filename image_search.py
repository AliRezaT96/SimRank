# https://github.com/openai/CLIP

import torch
import torch.nn.functional as F
from torch import nn

from PIL import Image
import time
import os
import pickle
from tqdm import tqdm
import random
import numpy as np
import heapq
from operator import itemgetter

import matplotlib.pyplot as plt

from model import create_model_and_transforms, get_tokenizer, ExtraLayer

SAVE_RESULTS = False
EXTRA_LAYER = True
DATA_SPLIT = 'train'
CACHE_DIR = 'saved_models/cache/'
MODEL_PATH = 'saved_models/RN50_last_layer_3.pt'

print("Loading model...")
device = "cuda" if torch.cuda.is_available() else "cpu"
model, _, preprocess = create_model_and_transforms('RN50', pretrained='openai', device=device, cache_dir=CACHE_DIR)
tokenizer = get_tokenizer('ViT-B-32-quickgelu', cache_dir=CACHE_DIR)


image_path = 'dataset/images/{}'.format(DATA_SPLIT)
image_fnames = os.listdir(image_path)

print("Loading features...")
with open("dataset/features_{}.obj".format(DATA_SPLIT), "rb") as f:
    features = pickle.load(f)


if not os.path.exists('logs'):
    os.makedirs('logs')
if not os.path.exists('logs/search_results'):
    os.makedirs('logs/search_results')

if EXTRA_LAYER:
    last_layer_model = ExtraLayer(embed_dim=1024)
    state_dict = torch.load(MODEL_PATH)
    last_layer_model.load_state_dict(state_dict)
    last_layer_model.eval()
    
    loss = nn.CosineSimilarity()
    
    with torch.no_grad():
        image_features = torch.Tensor(features['image_features'], device=device)
        text_features = torch.Tensor(features['text_features'], device=device)
        print("Cosine before extra layer:", loss(image_features, text_features).mean().item())
        image_out, text_out = last_layer_model(image_features, text_features)
        print("Cosine after extra layer:", loss(image_out, text_out).mean().item())
        features['image_features'] = image_out.cpu().numpy()
        features['text_features'] = text_out.cpu().numpy()
        # exit()





K = 10
while True:
    inp_str = input("rand / *.jpg / text * / exit : ")
    if inp_str == "rand" or inp_str in image_fnames:
        if inp_str == "rand":
            x = random.randint(0, len(image_fnames))
            fname = image_fnames[x]
            print("Image", fname, "has been chosen")
        else :
            fname = inp_str
        image = preprocess(Image.open(os.path.join(image_path, fname))).to(device).unsqueeze(0)
        with torch.no_grad():
            if EXTRA_LAYER:
                image_features = model.encode_image(image)
                image_features = last_layer_model.encode_image(image_features).squeeze(0).cpu().numpy()
            else:
                image_features = model.encode_image(image).squeeze(0).cpu().numpy()
        
        image_features_normalized = image_features / np.linalg.norm(image_features, 2)
        features_normalized = features['image_features'] / np.linalg.norm(features['image_features'], 2, axis=1, keepdims=True)
        _scores = np.matmul(features_normalized, image_features_normalized).tolist()
        scores = zip(_scores, features['file_names'])
    elif inp_str.startswith("text"):
        search_text = inp_str.replace("text", "", 1).strip()
        text = tokenizer([search_text]).to(device)
        with torch.no_grad():
            if EXTRA_LAYER:
                text_features = model.encode_text(text)
                text_features = last_layer_model.encode_text(text_features).squeeze(0).cpu().numpy()
            else:
                text_features = model.encode_text(text).squeeze(0).cpu().numpy()
            text_features_normalized = text_features / np.linalg.norm(text_features, 2)
            features_normalized = features['image_features'] / np.linalg.norm(features['image_features'], 2, axis=1, keepdims=True)
            _scores = np.matmul(features_normalized, text_features_normalized).tolist()
            scores = zip(_scores, features['file_names'])
    elif inp_str == "exit":
        break
    else:
        print("Bad Input!")
        continue
    
    top_k = heapq.nlargest(K, scores, key=itemgetter(0))

    if inp_str.startswith("text"):
        orig_img = Image.new('RGB', (400, 600))
    else:
        orig_img = Image.open(os.path.join(image_path, fname))

    similar_imgs = [Image.open(os.path.join(image_path, sim_fname)) for _, sim_fname in top_k]
    fig, axs = plt.subplots(1, K + 1, figsize=(15,10))
    fig.tight_layout()
    plt.subplots_adjust(wspace=0, hspace=0)
    axs[0].imshow(orig_img)
    if inp_str.startswith("text"):
        plt.title("Search Text: " + search_text)
    axs[0].title.set_text('Original Image')
    for i in range(K):
        axs[i + 1].imshow(similar_imgs[i])
    for i in range(K+1):
        axs[i].set_axis_off()
    plt.show() 
    if SAVE_RESULTS:
        if inp_str.startswith("text"):
            fig.savefig('logs/search_results/sample_text_{}.png'.format(search_text[:min(20, len(search_text))]))
        else:
            fig.savefig('logs/search_results/sample_{}.png'.format(fname.split('.')[0]))
    
