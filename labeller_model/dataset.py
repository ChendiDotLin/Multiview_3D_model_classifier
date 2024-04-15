# import torch
# from torch.utils.data import Dataset, DataLoader
# import os
# import glob


# class TransformedMultiView3DDataset(Dataset):
#     def __init__(self, data_dir):
#         """
#         data_dir (string): Directory with all the .pt files containing the transformed images and labels.
#         """
#         self.data_files = glob.glob(os.path.join(data_dir, "*.pt"))

#     def __len__(self):
#         return len(self.data_files)

#     def __getitem__(self, idx):
#         data_path = self.data_files[idx]
#         imgs, labels = torch.load(data_path)
#         # print(labels)
#         # labels_values = [int(label) for label in labels]
#         # labels_tensor = torch.tensor(labels_values)
#         return imgs, labels
from torch.utils.data import Dataset
import lmdb
import pickle
from PIL import Image
import io


class LMDBDataset(Dataset):
    def __init__(self, lmdb_path, transform=None):
        self.lmdb_path = lmdb_path
        self.transform = transform
        self.db = lmdb.open(
            lmdb_path, readonly=True, lock=False, readahead=False, meminit=False
        )
        with self.db.begin(write=False) as txn:
            self.length = txn.stat()["entries"]

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        with self.db.begin(write=False) as txn:
            key = f"key_{index}".encode("ascii")
            data = txn.get(key)
        image_data, label = pickle.loads(data)
        image = Image.open(io.BytesIO(image_data))

        if self.transform:
            image = self.transform(image)

        return image, label

    def __del__(self):
        self.db.close()
