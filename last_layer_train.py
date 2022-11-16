import os
import torch
import torch.nn.functional as F
from torch import nn

import numpy as np
from sklearn.linear_model import LogisticRegression
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR100
from tqdm import tqdm
import pickle
import time

from torch.utils.data import DataLoader
from utils.datamodel import ImageTextDataset, collate_batch

from model import create_model_and_transforms, ExtraLayer

EPOCHS = 20

# Load the model
device = "cuda" if torch.cuda.is_available() else "cpu"
# model, preprocess, _ = create_model_and_transforms("RN50", pretrained='openai', device=device)

# Load the dataset
# root = os.path.expanduser("~/.cache")
# train = CIFAR100(root, download=True, train=True, transform=preprocess)
# test = CIFAR100(root, download=True, train=False, transform=preprocess)


print("Loading features...")
with open("dataset/features_train.obj", "rb") as f:
    train_data = pickle.load(f)

print("Loading features...")
with open("dataset/features_val.obj", "rb") as f:
    val_data = pickle.load(f)



train_dataset = ImageTextDataset(train_data['image_features'], train_data['text_features'], train_data['file_names'], train=True)
val_dataset = ImageTextDataset(val_data['image_features'], val_data['text_features'], val_data['file_names'], train=True)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, collate_fn=collate_batch)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, collate_fn=collate_batch)



model = ExtraLayer(embed_dim=1024)
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
# optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
# scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60, 80], gamma=0.1)
cos_sim = nn.CosineSimilarity()
log_softmax = nn.LogSoftmax(dim=1)

model.train()
for epoch in range(EPOCHS):
    train_loss, train_n, val_loss, val_pos, val_neg, val_n = 0, 0, 0, 0, 0, 0

    for batch in train_loader:
        image_features = torch.Tensor(batch[0], device=device)
        text_features = torch.Tensor(batch[1], device=device)
        neg_text_features = torch.Tensor(batch[2], device=device)
        optimizer.zero_grad()
        image_out = model.encode_image(image_features)
        text_out = model.encode_text(text_features)
        neg_text_out = model.encode_text(neg_text_features)
        pos = cos_sim(image_out, text_out)
        neg = cos_sim(image_out, neg_text_out)
        loss = - pos.mean() + neg.mean()
        # loss = - log_softmax(torch.stack([pos, neg], dim=1))[:,0].mean()
        # loss = torch.maximum(pos_loss, neg_loss).mean()
        # pos_loss = pos_loss.mean()
        # neg_loss = neg_loss.mean()
        train_loss += loss
        loss.backward()
        optimizer.step()
        # train_pos_loss += pos_loss
        # train_neg_loss += neg_loss
        train_n += 1
    # scheduler.step()

    for batch in val_loader:
        image_features = torch.Tensor(batch[0], device=device)
        text_features = torch.Tensor(batch[1], device=device)
        neg_text_features = torch.Tensor(batch[2], device=device)
        image_out = model.encode_image(image_features)
        text_out = model.encode_text(text_features)
        neg_text_out = model.encode_text(neg_text_features)
        pos = cos_sim(image_out, text_out)
        neg = cos_sim(image_out, neg_text_out)
        loss = - pos.mean() + neg.mean()
        # loss = - log_softmax(torch.stack([pos, neg], dim=1))[:,0].mean()
        # loss = pos_loss + lam * neg_loss
        val_pos += pos.mean()
        val_neg += neg.mean()
        val_loss += loss
        val_n += 1

    print("Epoch: {} | Train Loss: {:.4f}| Val Loss: {:.4f} | Val Cos_Sim Pos: {:.4f} | Val Cos_Sim Neg: {:.4f}".format(epoch, train_loss / train_n, val_loss / val_n, val_pos / val_n, val_neg / val_n))

# for batch in val_loader:
#     image_features = torch.Tensor(batch[0], device=device)
#     text_features = torch.Tensor(batch[1], device=device)
#     neg_text_features = torch.Tensor(batch[2], device=device)
#     image_out = model.encode_image(image_features)
#     text_out = model.encode_text(text_features)
#     neg_text_out = model.encode_text(neg_text_features)
#     print(image_features)
#     print(text_features)
#     print(neg_text_features)
#     print("||||||||||||||||||||||||||||||||||||||")
#     print(image_out)
#     print(text_out)
#     print(neg_text_out)
#     exit()
#     loss = lam * loss_fn(image_out, neg_text_out).mean() - loss_fn(image_out, text_out).mean()
#     val_loss += loss
#     val_n += 1

torch.save(model.state_dict(), "saved_models/RN50_last_layer_{}.pt".format(int(time.time())))