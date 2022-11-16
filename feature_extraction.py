# https://github.com/openai/CLIP

import torch
from PIL import Image
import time
import os
import pickle
from tqdm import tqdm
import random

from model import create_model_and_transforms, get_tokenizer

BATCH_SZ = 64
CACHE_DIR = 'saved_models/cache/'

print("Loading model...")
device = "cuda" if torch.cuda.is_available() else "cpu"
model, _, preprocess = create_model_and_transforms('RN50', pretrained='openai', device=device, cache_dir=CACHE_DIR)
tokenizer = get_tokenizer('ViT-B-32-quickgelu', cache_dir=CACHE_DIR)

for split in ['train', 'val']:
    image_path = 'dataset/images/{}/'.format(split)
    text_path = 'dataset/captions/{}/'.format(split)
    image_cache = []
    text_cache = []
    fname_cache = []
    image_dirs = os.listdir(image_path)
    images = []
    texts = []
    fnames = []
    print("Loading images...")
    for i, f in enumerate(tqdm(image_dirs)):
        if not os.path.isfile(os.path.join(image_path, f)):
            continue
        image = preprocess(Image.open(os.path.join(image_path, f))).to(device)
        with open(os.path.join(text_path, f.replace('jpg', 'txt'))) as txt_file:
            text = tokenizer([' '.join(txt_file.readlines()).strip()]).squeeze(0).to(device)
        image_cache.append(image)
        text_cache.append(text)
        fname_cache.append(f)
        if i == len(image_dirs) - 1:
            image_list = []
            text_list = []
            fname_list = []
            rand_inds = list(range(len(image_cache)))
            random.shuffle(rand_inds)
            for i in rand_inds:
                image_list.append(image_cache[i])
                text_list.append(text_cache[i])
                fname_list.append(fname_cache[i])
            images = torch.stack(image_list, dim=0)
            texts = torch.stack(text_list, dim=0)
            fnames = fname_list

    print("Extracting features...")
    st_t = time.time()
    image_features_list = []
    text_features_list = []
    for i in tqdm(range(len(images))):
        image = images[i].unsqueeze(0)
        text = texts[i].unsqueeze(0)
        with torch.no_grad():
            image_features = model.encode_image(image).squeeze(0)
            text_features = model.encode_text(text).squeeze(0)
            image_features_list.append(image_features)
            text_features_list.append(text_features)
    image_features = torch.stack(image_features_list, dim=0).cpu().numpy()
    text_features = torch.stack(text_features_list, dim=0).cpu().numpy()
    print(image_features.shape)
    result = {'image_features': image_features, 'text_features': text_features, 'file_names': fnames}

    en_t = time.time()
    print("Feature Extraction Time:", en_t - st_t)

    with open("dataset/features_{}.obj".format(split), "wb") as f:
        pickle.dump(result, f)

    # with open("features.obj", "rb") as f:
    #     result = pickle.load(f)
