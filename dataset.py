import torch
from torchvision import transforms
from torch.utils.data import DataLoader,Dataset
from typing import Tuple 
from PIL import Image
import polars as pl

class PneumoniaDataset(Dataset):
    def __init__(self,csv1:str,csv2:str):
        self.testdf = pl.read_csv("test.csv")
        self.traindf = pl.read_csv("train.csv")
        self.mode = mode
        if mode == "train":
            pass
        elif mode  == "test":
            pass
        else:
            print("mode not set")


    

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, str]:
        path = pass
        image = Image.open(path)
        tensor = transforms.ToTensor()(image)
        label = "PNEUMONIA" if "PNEUMONIA" in path else "NORMAL"

        return tensor, label, path
    