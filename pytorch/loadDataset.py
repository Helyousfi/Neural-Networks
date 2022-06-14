import numpy as np 
import torch
# Map-style datasets
class simpleDataset(torch.utils.data.Dataset):
    def __init__(self, size : int):
        self.size = size
    def __len__(self) -> int:
        return self.size
    def __getitem__(self, index : int):
        return dict(
            x = np.eye(3) * index,
            y = index,
        )
dataset = simpleDataset(17)
loader = torch.utils.data.DataLoader(
    dataset,
    batch_size = 4,
    shuffle = True,
)
for batch in loader:
    x = batch["x"]
    y = batch["y"]
    print(f"X shape : {x.shape} y : {y}")
