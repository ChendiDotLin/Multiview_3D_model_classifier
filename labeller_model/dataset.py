
import torch
from torch.utils.data import Dataset, DataLoader
import os
import glob

class TransformedMultiView3DDataset(Dataset):
    def __init__(self, data_dir):
        """
        data_dir (string): Directory with all the .pt files containing the transformed images and labels.
        """
        self.data_files = glob.glob(os.path.join(data_dir, "*.pt"))

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, idx):
        data_path = self.data_files[idx]
        imgs, labels = torch.load(data_path)
        return imgs, labels