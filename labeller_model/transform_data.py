import torch
from torchvision import transforms
import lmdb
import os
import glob
import pickle
from PIL import Image
import io
import csv
from tqdm import tqdm


def create_lmdb_dataset(source_folder, lmdb_path, all_labels):
    map_size = int(1e12)  # Size of the database, this is a maximum limit
    db = lmdb.open(lmdb_path, map_size=map_size)

    with db.begin(write=True) as txn:
        # Initialize progress bar with total steps equal to the number of items in labels
        pbar = tqdm(total=len(all_labels), desc="Creating LMDB Dataset", unit="item")
        for idx, (uid, labels) in enumerate(all_labels.items()):
            img_folder = os.path.join(source_folder, uid)
            # Check if the image folder exists
            if not os.path.exists(img_folder):
                # Handle the case where the image folder doesn't exist
                print(f"Image folder does not exist: {img_folder}")
                continue
            images = sorted(glob.glob(f"{img_folder}/*.png"))
            for img_path in images:
                img = Image.open(img_path)
                img_byte_arr = io.BytesIO()
                img.save(img_byte_arr, format="PNG")
                image_data = img_byte_arr.getvalue()

                key = f"{uid}_{os.path.basename(img_path)}".encode("ascii")
                labels_values = [int(label) for label in labels]
                labels_tensor = torch.tensor(labels_values)
                value = pickle.dumps((image_data, labels_tensor))
                txn.put(key, value)
            # Update the progress bar
            pbar.update(1)
        pbar.close()
    db.close()


def load_labels_from_csv(csv_file):
    labels = {}
    with open(csv_file, mode="r") as file:
        reader = csv.reader(file)
        next(reader)  # Skip the first line, which is the header
        for row in reader:
            uid = row[0]
            labels[uid] = tuple(row[1:])  # Adjust the slicing if necessary
    return labels


# Example usage:
csv_file_path = "training_label.csv"
labels = load_labels_from_csv(csv_file_path)
print("finish loading labels")

# Define your transform
transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)
print("finish transformation")

# Instantiate your dataset
create_lmdb_dataset(
    source_folder="views", lmdb_path="./transformed_data", all_labels=labels
)
print("finish creating lmdb")
# Assuming labels are stored as a tuple:
# (style, score, is_multi_object, is_weird, is_scene, is_figure, is_transparent, density)
