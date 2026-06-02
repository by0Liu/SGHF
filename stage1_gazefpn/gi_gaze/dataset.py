import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import tifffile as tiff
import torchvision.transforms.functional as f
from numpy.random import randint
from torchvision.transforms import InterpolationMode as IM

h_size = 512
w_size = 512

resize = transforms.Resize([h_size, w_size],
                           interpolation= transforms.InterpolationMode.BICUBIC,
                           antialias=True)
to_Tensor = transforms.Compose([
    transforms.PILToTensor(),
    transforms.ConvertImageDtype(torch.float32),
])


def read_gaze_tif_to_tensor(path):

    g = tiff.imread(path)
    if g.ndim == 3 and g.shape[2] > 1:
        g = g.mean(axis=2)
    g = g.astype(np.float32)
    mn, mx = g.min(), g.max()
    if mx > mn:
        g = (g - mn) / (mx - mn)
    else:
        g[:] = 0.0
    return torch.from_numpy(g)[None, ...]


#benign = class index 0, malignant = 1
class GazeDataset(Dataset):
    def __init__(self,
                 folder_path,
                 transform= None,
                 load_gaze = False
                 ):
        self.images = []
        self.gaze_images = []
        self.label = []
        self.transform = transform
        benign_path = folder_path+'Type 1/'
        for file in os.listdir(benign_path+'images'):
            img = Image.open(benign_path+'images/'+file)
            img = resize(to_Tensor(img))
            self.images.append(img)
            if load_gaze:
                # img_gaze = Image.open(benign_path + 'attentions/' + file.replace('.png', '.tif'))
                # img_gaze = resize(to_Tensor(img_gaze))c
                img_gaze = read_gaze_tif_to_tensor(benign_path + 'attentions/' + file.replace('.png', '.tif'))
                img_gaze = resize(img_gaze)

                self.gaze_images.append(img_gaze)
            else:
                self.gaze_images.append(torch.zeros([1, h_size, w_size]))
            self.label.append(0)
        malignant_path = folder_path + 'Type 2/'
        for file in os.listdir(malignant_path + 'images'):
            img = Image.open(malignant_path + 'images/' + file)
            img = resize(to_Tensor(img))
            self.images.append(img)
            if load_gaze:
                # img_gaze = Image.open(malignant_path + 'attentions/' + file.replace('.png', '.tif'))
                # img_gaze = resize(to_Tensor(img_gaze))
                img_gaze = read_gaze_tif_to_tensor(benign_path + 'attentions/' + file.replace('.png', '.tif'))
                img_gaze = resize(img_gaze)
                self.gaze_images.append(img_gaze)
            else:
                self.gaze_images.append(torch.zeros([1, h_size, w_size]))
            self.label.append(1)
        type_path = folder_path + 'Type 3/'
        for file in os.listdir(type_path + 'images'):
            img = Image.open(type_path + 'images/' + file)
            img = resize(to_Tensor(img))
            self.images.append(img)
            if load_gaze:
                # img_gaze = Image.open(type_path + 'attentions/' + file.replace('.png', '.tif'))
                # img_gaze = resize(to_Tensor(img_gaze))
                img_gaze = read_gaze_tif_to_tensor(benign_path + 'attentions/' + file.replace('.png', '.tif'))
                img_gaze = resize(img_gaze)
                self.gaze_images.append(img_gaze)
            else:
                self.gaze_images.append(torch.zeros([1, h_size, w_size]))
            self.label.append(2)
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        gaze = self.gaze_images[idx]
        if self.transform:
            img, gaze = self.transform(img, gaze)
        label = self.label[idx]
        return img, label, gaze

def rand_flip_rotate_augmentation(img, mask):
    if randint(0, 2):
        angle = int(randint(0, 360))
        img  = f.rotate(img, angle, interpolation=IM.BILINEAR, fill=0)
        mask = f.rotate(mask, angle, interpolation=IM.BILINEAR, fill=0)
    if randint(0, 2):
        img  = f.hflip(img)
        mask = f.hflip(mask)
    if randint(0, 2):
        img  = f.vflip(img)
        mask = f.vflip(mask)
    return img, mask


class image_batch:
    def __init__(self, data):
        transposed_data = list(zip(*data))
        self.img = torch.stack(transposed_data[0], 0)
        self.label = torch.tensor(transposed_data[1], dtype=torch.int64)
        self.gaze = torch.stack(transposed_data[2], 0)
        self.size = self.img.size(0)

    # custom memory pinning method on custom type
    def pin_memory(self):
        self.img = self.img.pin_memory()
        self.gaze = self.gaze.pin_memory()
        return self

def image_batch_wrapper(batch):
    return image_batch(batch)

if __name__ == '__main__':
    dataset = GazeDataset(
        folder_path= 'data/test/',
        transform=rand_flip_rotate_augmentation,
        load_gaze= True
    )
    test_loader = DataLoader(dataset, batch_size=3, shuffle=True, collate_fn=image_batch_wrapper, pin_memory=True)
    print(len(dataset), len(test_loader))
    for batch_ndx, batch in enumerate(test_loader):
        print(batch_ndx, batch.size, batch.img.shape, batch.gaze.shape, batch.label)

