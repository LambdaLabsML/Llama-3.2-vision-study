import csv
import os
from typing import Dict

import numpy as np
from torch.utils.data import Dataset

class RandomImageDataset(Dataset):
    def __init__(self, num_images=100, image_size=(256, 256)):
        self.num_images = num_images
        self.image_size = image_size

    def __len__(self):
        return self.num_images

    def __getitem__(self, idx):
        return np.random.randint(0, 256, size=(3, *self.image_size), dtype=np.uint8)

def append_dict_to_csv(csv_filepath, data: Dict[str, str]):
    assert len(data.keys()) > 0, "Data dictionary must not be empty"
    fieldnames = list(data.keys())
    file_exists = os.path.exists(csv_filepath)
    with open(csv_filepath, mode='a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(data)

