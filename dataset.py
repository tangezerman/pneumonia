import torch
from torchvision import transforms
from torch.utils.data import DataLoader,Dataset
from typing import Tuple 
from PIL import Image
import polars as pl


class PneumoniaDataset(Dataset):
    def __init__(self,csv: str,mode: str):
        self.df = pl.read_csv(csv)
        self.testdf = self.df.select("test")
        self.traindf = self.df.select("train")
        self.mode = mode
        if mode == "train":
            self.df = self.traindf
            self.column = "train"
        elif mode  == "test":
            self.df = self.testdf
            self.column = "test"
        else:
            print("mode not set")


    

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, str, str]:
        path = self.df[index,self.column ]
        image = Image.open(path)
        tensor = transforms.ToTensor()(image)
        label = "PNEUMONIA" if "PNEUMONIA" in path else "NORMAL"

        return tensor, label, path
    
    def __len__(self) -> int:
        return len(self.df)