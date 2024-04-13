import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os
import glob
from PIL import Image
import csv


class MultiView3DDataset(Dataset):
    def __init__(self, root_dir, labels, transform=None):
        """
        root_dir (string): Directory with all the folders containing images for each uid.
        labels (dict): A dictionary where keys are uids and values are the labels for the 5 attributes.
        transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.labels = labels
        self.transform = transform
        self.uids = list(labels.keys())  # List of all uids

    def __len__(self):
        return len(self.uids)

    def __getitem__(self, idx):
        uid = self.uids[idx]
        img_folder = os.path.join(self.root_dir, uid)
        images = sorted(glob.glob(f"{img_folder}/*.jpg"))  # Assuming images are .jpg
        imgs = []
        for img_path in images:
            img = Image.open(img_path).convert("RGB")
            if self.transform:
                img = self.transform(img)
            imgs.append(img)
        imgs_tensor = torch.stack(imgs)

        # Assuming labels are stored as a tuple:
        # (style, score, is_multi_object, is_weird, is_scene, is_figure, is_transparent, density)
        labels = self.labels[uid]
        return imgs_tensor, labels


# Assuming the MultiView3DDataset is already defined and instantiated as `dataset`
# And you have defined your `transform` as before
def save_transformed_dataset(dataset, save_dir):
    for idx, (imgs, labels) in enumerate(dataset):
        uid = dataset.uids[idx]  # Get the UID for the current item
        save_path = os.path.join(save_dir, f"{uid}.pt")  # Define the save path
        # Save both images and labels together
        torch.save((imgs, labels), save_path)
        print(f"Saved transformed data for UID {uid} to {save_path}")


def load_labels_from_csv(csv_file):
    labels = {}
    with open(csv_file, mode="r") as file:
        reader = csv.reader(file)
        next(reader)  # Skip the first line, which is the header
        for row in reader:
            uid = row[0]
            labels[uid] = tuple(row[1:])  # Adjust the slicing if necessary
            print(labels)
    return labels


# Example usage:
csv_file_path = "training_label.csv"
labels = load_labels_from_csv(csv_file_path)

# Define your transform
transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

# Instantiate your dataset
dataset = MultiView3DDataset(root_dir="views", labels=labels, transform=transform)
# Instantiate DataLoader
train_loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4)

# Specify the directory where you want to save the transformed dataset
save_dir = "./transformed_data"
save_transformed_dataset(dataset, save_dir)


# Assuming labels are stored as a tuple:
# (style, score, is_multi_object, is_weird, is_scene, is_figure, is_transparent, density)
