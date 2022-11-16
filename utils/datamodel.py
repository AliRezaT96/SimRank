from torch.utils.data import Dataset
import numpy as np

class ImageTextDataset(Dataset):
    def __init__(self, images, texts, fnames, train=True):
        self.images = images
        self.texts = texts
        self.fnames = fnames
        self.train = train
    def __len__(self):
        return len(self.fnames)
    def __getitem__(self, idx):
        image = self.images[idx]
        text = self.texts[idx]
        if self.train:
            neg_text = self.texts[np.random.randint(0, len(self.texts))] # self.texts[(idx + 1) % len(self.texts)] 
            sample = {"image": image, "text": text, "neg_text": neg_text}
            return sample
        sample = {"image": image, "text": text}
        return sample

def collate_batch(batch):
    image_list, text_list = [], []
    train = 'neg_text' in batch[0].keys()
    if train:
        neg_text_list = []
    for sample in batch:
        image = sample['image']
        text = sample['text']
        image_list.append(image)
        text_list.append(text)
        if train:
            neg_text = sample['neg_text']
            neg_text_list.append(neg_text)
    images = np.stack(image_list)
    texts = np.stack(text_list)
    if train:
        neg_texts = np.stack(neg_text_list)
        return images, texts, neg_texts
    return images, texts