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
import torch


class LMDBDataset(Dataset):
    def __init__(self, lmdb_path, transform=None):
        self.lmdb_path = lmdb_path
        # self.uids = uids
        self.transform = transform
        # self.env = lmdb.open(
        #     self.lmdb_path,
        #     readonly=True,
        #     max_readers=64,
        #     lock=False,
        #     readahead=False,
        #     meminit=False,
        # )
        # with self.env.begin(write=False) as txn:
        #     self.length = txn.stat()["entries"]
        # self.env = lmdb.open(self.lmdb_path, readonly=True, lock=False)
        self.uids = self._load_uids()

    def _load_uids(self):
        uids = []
        with lmdb.open(self.lmdb_path, readonly=True, lock=False) as env:
            with env.begin(write=False) as txn:
                cursor = txn.cursor()
                for key, _ in cursor:
                    uids.append(
                        key.decode("utf-8")
                    )  # assuming keys are bytes, decode as needed
        return uids

    def __len__(self):
        # return self.length
        with lmdb.open(self.lmdb_path, readonly=True, lock=False) as env:
            with env.begin(write=False) as txn:
                return txn.stat()["entries"]

    def __getitem__(self, index):
        uid = self.uids[index]  # Get the UID corresponding to the index
        # with self.env.begin(write=False) as txn:
        #     # Assuming keys are stored as bytes, and index starts at 0
        #     key = uid.encode("ascii")
        #     value = txn.get(key)

        #     if value is None:
        #         print(
        #             f"Data not found for key {key}. Check key format and data integrity in the LMDB database."
        #         )

        #     image_data_list, label = pickle.loads(value)
        #     images = [
        #         Image.open(io.BytesIO(img_data)).convert("RGB")
        #         for img_data in image_data_list
        #     ]

        #     if self.transform:
        #         images = [self.transform(img) for img in images]

        #     imgs_tensor = torch.stack(images)  # Stack images into a tensor
        #     return imgs_tensor, label
        with lmdb.open(self.lmdb_path, readonly=True, lock=False) as env:
            with env.begin(write=False) as txn:
                key = f"{uid}".encode("ascii")
                value = txn.get(key)
        if not value:
            raise ValueError("Could not retrieve data - key not found in LMDB.")

        image_data_list, label, metadata = pickle.loads(value)
        images = [
            Image.open(io.BytesIO(img_data)).convert("RGB")
            for img_data in image_data_list
        ]
        if self.transform:
            images = [self.transform(img) for img in images]
        imgs_tensor = torch.stack(images)
        if len(label) != 8:
            print("uid: ", uid)
        return imgs_tensor, label, metadata

    def __del__(self):
        if hasattr(self, "env") and self.env is not None:
            self.env.close()
