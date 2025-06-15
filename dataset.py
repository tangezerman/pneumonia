import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from typing import Tuple
from PIL import Image
import polars as pl


class PneumoniaDataset(Dataset):
    def __init__(self, csv: str, mode: str):
        self.df = pl.read_csv(csv)
        self.mode = mode

        if mode == "train":
            self.data = self.df.select("train").to_series().drop_nulls().to_list()

        elif mode == "test":
            self.data = self.df.select("test").to_series().drop_nulls().to_list()

        else:
            print("mode not set")

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, str, str]:
        path = self.data[index]

        if self.mode == "train":
            path = self.data[index]

        elif self.mode == "test":
            path = self.data[index]
            
        else:
            print("mode not set - getitem()")
        image = Image.open(path)
        transform = transforms.Compose(
            [
                transforms.Resize((1024, 1024)),
                transforms.Grayscale(num_output_channels=1),
                transforms.ToTensor(),
            ]
        )
        tensor = transform(image)
        if "pneumonia" in path.lower():
            if "bacteria" in path.lower():
                label = "bacteria"

            elif "virus" in path.lower():
                label = "virus"
                
            else:
                raise ValueError((f"Unknown pneumonia type in path: {path}"))
        else:
            label = "normal"

        return tensor, label, path

    def __len__(self) -> int:
        return len(self.data)


if __name__ == "__main__":
    dataset = PneumoniaDataset("dataset.csv", "train")
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4)

    dataiter = iter(dataloader)
    data = next(dataiter)
    tensor, label, path = data
    print(tensor, label, path)
