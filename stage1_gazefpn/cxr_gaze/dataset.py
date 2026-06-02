import pandas as pd
import imageio
import numpy as np
from PIL import Image
from torchvision import transforms
import torch
from torch.utils.data import Dataset,DataLoader


labels = ["Atelectasis","Cardiomegaly","Consolidation","Edema","Enlarged Cardiomediastinum","Fracture","Lung Lesion",
          "Lung Opacity","No Finding","Pleural Effusion","Pleural Other","Pneumonia","Pneumothorax","Support Devices"]

IMAGE_ROOT_PREFIX = "/path/to/mimic-cxr-jpg/files/"

class MIMICDataset(Dataset):
    def __init__(self,csv_path,transforms = None):
        self.data = pd.read_csv(csv_path)
        self.labels = (self.data[labels].astype('float').values > 0) * 1.
        self.transforms = transforms

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if idx >= len(self):
            raise StopIteration
        filepath = self.data.iloc[idx]["image"]
        img = imageio.imread(filepath)
        filepath = filepath.replace("\\", "/")
        filepath = filepath.replace(IMAGE_ROOT_PREFIX, "")
        filepath = filepath.replace("/", "")
        filepath = filepath.replace("jpg", "dcm")
        filepath = filepath.replace("zhouxinjiongmimic", "")
        img = Image.fromarray(img)
        img = img.convert("RGB")

        if self.transforms is not None:
            img = self.transforms(img)

        return img, np.array(self.labels[idx]).astype(np.float32), filepath
